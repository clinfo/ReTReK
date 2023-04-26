import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdChemReactions
import torch
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
from argparse import ArgumentParser
from model.in_scope_filter import InScopeModel

def get_count_fgp(mol, n_bits=2048, radius=2):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    Chem.SanitizeMol(mol)
    bit_info = {}
    fgp = np.zeros(n_bits)
    AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits, bitInfo=bit_info)
    for bit_id, active in bit_info.items():
        fgp[bit_id] = len(active)
    return fgp


def get_reaction_count_fgp(reaction, n_bits=2048, radius=2):
    reactants = reaction.split(">")[0]
    product = reaction.split(">")[-1]
    p_fgp = get_count_fgp(product, n_bits, radius)
    r_fgp = get_count_fgp(reactants, n_bits, radius)
    return p_fgp - r_fgp


class ReactionDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, reaction_column, product_column, label_column):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.reaction_column = reaction_column
        self.product_column = product_column
        self.label_column = label_column
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        reaction = torch.FloatTensor(get_reaction_count_fgp(row[self.reaction_column], n_bits=2048))
        product = torch.log(torch.FloatTensor(get_count_fgp(row[self.product_column], n_bits=16384)) + 1)
        y = torch.FloatTensor([row[self.label_column]])
        return reaction, product, y


class Meter:
    def __init__(self, coeff=0.95):
        self.coeff = coeff
        self.value = np.nan
    
    def update(self, value):
        if not self.value == self.value:
            self.value = value
        else:
            self.value = self.coeff * self.value + (1-self.coeff) * value


def prepare_batch(batch, device):
    reaction, product, y = batch
    reaction = reaction.to(device)
    product = product.to(device)
    y = y.to(device)
    return reaction, product, y


def train_epoch(model, optimizer, loss_fn, loader, epoch, acc_meter, auc_meter, device):
    pbar = tqdm(loader)
    model.train()
    auc = 0
    for i, batch in enumerate(pbar):
        reaction, product, y = prepare_batch(batch, device)

        out = model(reaction, product, logits=True)
        loss = loss_fn(out, y)
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%100 == 0:
            # Metrics computation
            labels = y.cpu().numpy()
            preds = torch.sigmoid(out).detach().cpu().numpy()
            if not (labels==labels[0]).all():
                auc = roc_auc_score(labels, preds)
            bin_preds = (preds > 0.5).astype(int)
            acc = accuracy_score(labels, bin_preds)
            acc_meter.update(acc)
            auc_meter.update(auc)
        pbar.set_description(f"Epoch: {epoch} | Loss: {loss:.4f} | Accuracy: {acc_meter.value:.4f} | AUC: {auc_meter.value:.4f}")


def validation_epoch(model, loader, device):
    pbar = tqdm(loader, leave=False)
    model.eval()
    all_outs = []
    all_labels = []
    with torch.no_grad():
        for batch in pbar:
            reaction, product, y = prepare_batch(batch, device)
            out = model(reaction, product)
            all_outs.append(out.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    labels = np.concatenate(all_labels, axis=0)
    preds = np.concatenate(all_outs, axis=0)
    
    auc = roc_auc_score(labels, preds)
    acc = accuracy_score(labels, (preds>0.5).astype(int))
    
    print(f"Validation metrics: Acc: {acc:.4f} | AUC: {auc:.4f}")
    return auc, acc

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--csv-path", type=str, help="Path to csv dataset")
    parser.add_argument("-rc", "--reaction-column", type=str, help="Reaction column name", default="reaction_smiles")
    parser.add_argument("-pc", "--product-column", type=str, help="Product column name", default="main_product")
    parser.add_argument("-lc", "--label-column", type=str, help="Label column name", default="label")
    parser.add_argument("-pw", "--pos-weight", type=float, help="Positive weight for BCE loss", default=1.)
    parser.add_argument("-lr", "--learning-rate", type=float, help="Learning rate", default=1e-4)
    parser.add_argument("-wd", "--weight-decay", type=float, help="Weight decay", default=1e-3)
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs", default=5)
    parser.add_argument("-nw", "--num-workers", type=int, help="Number of workers for dataloaders", default=4)
    parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("-sp", "--save-path", type=str, help="Save folder path", default="results")
    args = parser.parse_args()
    return args

def main(args):

    ds = ReactionDataset(
        csv_path=args.csv_path,
        reaction_column=args.reaction_column,
        product_column=args.product_column,
        label_column=args.label_column
    )

    indices = np.arange(len(ds))
    np.random.shuffle(indices)
    train_ds = torch.utils.data.Subset(ds, indices=indices[:int(0.8*len(ds))])
    val_ds = torch.utils.data.Subset(ds, indices=indices[int(0.8*len(ds)):])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=True, num_workers=args.num_workers)

    model = InScopeModel()
    device = torch.device("cuda") if args.use_cuda else torch.device("cpu")
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([args.pos_weight]).to(device))
    save_path = Path(args.save_path) / "best_ckpt.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model.to(device)
    acc_meter = Meter()
    auc_meter = Meter()
    epochs = args.epochs
    best_auc = -np.inf

    for epoch in range(1, epochs+1):
        train_epoch(model, optimizer, loss_fn, train_loader, epoch, acc_meter, auc_meter, device)
        auc, acc = validation_epoch(model, val_loader, device)
        if auc > best_auc:
            print("New best checkpoint!")
            torch.save(model.state_dict(), save_path)
            best_auc = auc

if __name__ == "__main__":
    args = parse_args()
    main(args)
