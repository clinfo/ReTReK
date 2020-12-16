import argparse
import os

from bs4 import BeautifulSoup
from rdkit import Chem
from rdkit.Chem import AllChem, Draw


def get_parser():
    parser = argparse.ArgumentParser(
        description='description',
        usage='ex) python visualize.py -r example1/route01/ -o output.html'
    )
    parser.add_argument(
        '-r', '--result_dir', action='store', default=None, required=True,
        help='help'
    )
    parser.add_argument(
        '-o', "--output", action="store", default="output.html",
        help="help"
    )
    return parser.parse_args()


def create_images(result_dir, states_lines, reaction_lines):
    # State
    for i, sl in enumerate(states_lines):
        mols = []
        for m in sl.split('.'):
            mol = Chem.MolFromSmiles(m)
            Chem.SanitizeMol(mol)
            mols.append(mol)
        #mols = [Chem.SanitizeMol(Chem.MolFromSmarts(m)) for m in sl.split('.')]
        img = Draw.MolsToGridImage(mols, subImgSize=(200, 200), molsPerRow=len(mols))
        img.save(os.path.join(result_dir, f"states{i:02d}.png"))
    # Reaction
    for i, rl in enumerate(reaction_lines):
        rxns = AllChem.ReactionFromSmarts(rl)
        img = Draw.ReactionToImage(rxns, subImgSize=(80, 80), useSVG=True)
        with open(os.path.join(result_dir, f"reaction{i:02d}.svg"), 'w') as f:
            f.write(img)


def create_html_file(result_dir, state_num, reaction_num, output):
    with open("template.html", mode="rt", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    # Title
    title = soup.new_tag("title")
    title.string = result_dir
    soup.find("head").append(title)
    # Header
    soup.find("h1").string = result_dir

    # State
    for i in range(state_num):
        section_mol = soup.new_tag("section", attrs={"class": "mol"})
        h3_num = soup.new_tag("h3", attrs={"class": "number"})
        h3_num.string = f"#{i}"
        img_state = soup.new_tag("img",
                                 attrs={"class": "image",
                                        "src": os.path.join("./", f"states{i:02}.png"),
                                        "alt": os.path.join("./", f"states{i:02}.png")}
                                 )
        section_mol.append(h3_num)
        section_mol.append(img_state)
        soup.find("div", attrs={"class": "state"}).append(section_mol)
    div_bottom = soup.new_tag("div", attrs={"class": "bottom"})
    soup.find("div", attrs={"class": "state"}).append(div_bottom)
    # Reaction
    for i in range(reaction_num):
        section_template = soup.new_tag("section", attrs={"class": "template"})
        h3_num = soup.new_tag("h3", attrs={"class": "number"})
        h3_num.string = f"#{i}"
        img_reaction = soup.new_tag("img",
                                    attrs={"class": "image",
                                           "src": os.path.join("./", f"reaction{i:02}.svg"),
                                           "alt": os.path.join("./", f"reaction{i:02}.svg")})
        section_template.append(h3_num)
        section_template.append(img_reaction)
        soup.find("div", attrs={"class": "reaction"}).append(section_template)
    div_bottom = soup.new_tag("div", attrs={"class": "bottom"})
    soup.find("div", attrs={"class": "reaction"}).append(div_bottom)
    with open(os.path.join(result_dir, output), 'w') as f:
        f.write(str(soup))


def main():
    args = get_parser()
    result_dir = args.result_dir
    with open(os.path.join(result_dir, "state.sma") , 'r') as f1, open(os.path.join(result_dir, "reaction.sma"), 'r') as f2:
        states_lines = [l.strip("\n") for l in f1.readlines()]
        reaction_lines = [l.strip("\n") for l in f2.readlines()]
    create_images(result_dir, states_lines, reaction_lines)
    create_html_file(result_dir, len(states_lines), len(reaction_lines), args.output)


if __name__ == "__main__":
    main()
