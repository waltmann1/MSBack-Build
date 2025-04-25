from msback.MSToolProtein import MSToolProtein
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


class Ligand(MSToolProtein):
    def __init__(self, pdbfile, yaml=None, weights_path=None, smiles=None):
        super(Ligand, self).__init__(pdbfile=pdbfile)

        self.map = None
        self.resmap = None
        self.yaml = yaml

        self.chroma_weights_path = weights_path
        if weights_path is None:
            self.chroma_weights_path = self.get_local_weights_path()

        if yaml is not None:
            self.map = self.read_cg_yaml(yaml)
            self.resmap = self.res_index_map()
        self.rdkit_molecule = self.get_rdkit_molecule(smiles=smiles)

    def get_rdkit_molecule(self, smiles=None):
        name = self.name + ".pdb"
        import os
        if not os.path.exists(name):
            self.write(name)
        m = Chem.MolFromPDBFile(name)
        if smiles is not None:
            m = Chem.MolFromSmiles(smiles)

        m = Chem.AddHs(m)
        AllChem.EmbedMolecule(m)
        AllChem.ComputeGasteigerCharges(m)
        return m

    def get_bonds_charges_types_names(self):
        m = self.rdkit_molecule
        bonds = []
        charges = []
        types = []
        names = []
        AllChem.ComputeGasteigerCharges(self.rdkit_molecule)
        for atom in m.GetAtoms():
            types.append(atom.GetSymbol())
            charge = atom.GetDoubleProp("_GasteigerCharge")
            if not np.isnan(charge):
                charges.append(charge)
            else:
                charges.append(0)
            names.append(types[-1] + str(types.count(types[-1])))

        for bond in m.GetBonds():
            bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        return bonds, charges, types,names
        
    def write_ff(self):
        bonds,charges,types,names= self.get_bonds_charges_types_names()
        space = "                        "
        resid = self.name
        xml_name = self.name + "_onesite_ff.xml"
        with open(xml_name, "w") as f:
            f.write("<Forcefield>\n")
            f.write("        <Residues>\n")
            f.write('                <Residue name="' + str(resid)  + '">\n')
            for ind,charge in enumerate(charges):
                f.write(space + '<Atom charge="' + str(self.fix_charge(charge)) +
                '" name="'+ str(names[ind]) + '"  type="' + str(types[ind][0])   + '"/>\n')
            for ind,bond in enumerate(bonds):
                f.write(space + '<Bond atomName1="' + str(names[bond[0]]) +
                '" atomName2="'+ str(names[bond[1]])    + '"/>\n')
            f.write("                </Residue>\n")
            f.write("        </Residues>\n")
            f.write("</Forcefield>\n")
        return xml_name


    def fix_charge(self, charge):
        pos = charge > 0
        charge= str(round(charge, 2))
        if pos:
            charge = "+" + str(charge)
        while len(charge) < 5:
            charge = charge + " "
        #print(charge, "now")
        return charge

class IP6(Ligand):
    def __init__(self, pdbfile):
        super(IP6, self).__init__(pdbfile, smiles="O=P(O)([O-])OC1C(OP(=O)(O)[O-])C(OP(=O)(O)[O-])C(OP(=O)(O)[O-])C(OP(=O)(O)[O-])C1OP(=O)(O)[O-]")





    
