import h5py
import numpy as np

# Function to explore and load H5 file
def explore_h5_file(file_path):
    data = []
    with h5py.File(file_path, 'r') as file:
        file.visititems(lambda name, obj: data.extend(obj[:]) if isinstance(obj, h5py.Dataset) else None)
    return np.array(data)

patient_embeddings = explore_h5_file('../preprocesseddata/G_patients.h5py')
disease_embeddings = explore_h5_file('../preprocesseddata/G_diseases.h5py')

# Function to compute cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Predict cancer type for each patient
predictions = []
for i, patient in enumerate(patient_embeddings):
    similarities = [cosine_similarity(patient, disease) for disease in disease_embeddings]
    predicted_cancer_type = np.argmax(similarities)  # Get most similar disease type
    predictions.append(f"Patient_{i+1}: Cancer_Type_{predicted_cancer_type}")

# Save predictions to a text file
with open("../results/predictions.txt", "w") as f:
    f.write("\n".join(predictions))

print("Predictions saved to predictions.txt")
