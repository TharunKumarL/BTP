import h5py
import numpy as np
def explore_h5_file(file_path):
    data = []
    with h5py.File(file_path, 'r') as file:
        file.visititems(lambda name, obj: data.extend(obj[:]) if isinstance(obj, h5py.Dataset) else None)
    return np.array(data)
patient_embeddings = explore_h5_file('../preprocesseddata/G_patients.h5py')
gene_embeddings = explore_h5_file('../preprocesseddata/G_genes.h5py')
gene_complexes = explore_h5_file('../preprocesseddata/G_complexes.h5py')
drugs = explore_h5_file('../preprocesseddata/G_drugs.h5py')
pathways = explore_h5_file('../preprocesseddata/G_pathways.h5py')
disease_embeddings = explore_h5_file('../preprocesseddata/G_diseases.h5py')
d2p=explore_h5_file('../matrices/d2p.h5py') 
print(d2p)
print(disease_embeddings.shape)
print(pathways.shape)
print(drugs.shape)
print(gene_complexes.shape)
print(gene_embeddings.shape)
print(patient_embeddings.shape)
import numpy as np

data = np.load("drugcentral.npy")
print(data.shape)
