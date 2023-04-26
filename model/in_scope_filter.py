import torch

class HighwayNetwork(torch.nn.Module):
    def __init__(self, dimension, n_layers, activation):
        super().__init__()
        self.n_layers = n_layers
        self.transform = torch.nn.ModuleList([torch.nn.Linear(dimension, dimension) for _ in range(self.n_layers)])
        self.gate = torch.nn.ModuleList([torch.nn.Linear(dimension, dimension) for _ in range(self.n_layers)])
        self.activation = activation
    
    def forward(self, x):
        for layer in range(self.n_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            h = self.activation(self.transform[layer](x))
            x = h * gate + x * (1-gate)
        return x
            
class InScopeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.product_network = torch.nn.Sequential(
            torch.nn.Linear(16384, 1024),
            torch.nn.ELU(),
            torch.nn.Dropout(0.3),
            HighwayNetwork(1024, 5, torch.nn.functional.elu)
        )
        self.reaction_layer = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.ELU()
        )
        self.cosine_sim = torch.nn.CosineSimilarity(dim=1)
    
    def forward(self, reaction, product, logits=False):
        r = self.reaction_layer(reaction)
        p = self.product_network(product)
        sim = self.cosine_sim(p, r).view(-1, 1)
        out = 10*sim # necessary for sigmoid to cover the range 0-1.
        if logits:
            return out
        else:
            return torch.sigmoid(out)
