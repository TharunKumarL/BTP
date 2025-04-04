import os
import deepchem as dc
from rdkit import Chem
import hickle as hkl

# Paths
drug_smiles_file = '../data/223drugs_pubchem_smiles.txt'
save_dir = '../data/GDSC/drug_graph_feat'

# Read SMILES
with open(drug_smiles_file, 'r') as f:
    pubchemid2smile = {
        line.strip().split('\t')[0]: line.strip().split('\t')[1]
        for line in f if line.strip()
    }

# Create directory if not exists
os.makedirs(save_dir, exist_ok=True)

# Initialize featurizer once
featurizer = dc.feat.graph_features.ConvMolFeaturizer()

# Process each molecule
for pubchem_id, smile in pubchemid2smile.items():
    print(f"Processing PubChem ID: {pubchem_id}")
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        print(f"Invalid SMILES for PubChem ID {pubchem_id}")
        continue
    mol_object = featurizer.featurize([mol])
    if not mol_object or mol_object[0] is None:
        print(f"Featurization failed for {pubchem_id}")
        continue

    features = mol_object[0].atom_features
    degree_list = mol_object[0].deg_list
    adj_list = mol_object[0].canon_adj_list

    # Save features
    out_path = os.path.join(save_dir, f'{pubchem_id}.hkl')
    hkl.dump([features, adj_list, degree_list], out_path)
