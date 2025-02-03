import numpy as np
import h5py
from tensorflow.keras.models import load_model

def explore_h5_file(file_path):
    data = []
    with h5py.File(file_path, 'r') as file:
        file.visititems(lambda name, obj: data.extend(obj[:]) if isinstance(obj, h5py.Dataset) else None)
    return np.array(data)

# Load embeddings
patient_embeddings = explore_h5_file('../preprocesseddata/G_patients.h5py')
disease_embeddings = explore_h5_file('../preprocesseddata/G_diseases.h5py')

# Use a generic cancer embedding for prediction
# Replace `0` with the index of a suitable default cancer type if available
default_cancer_embedding = disease_embeddings[0]

# Load the model
model = load_model('../models/patient_cancer_model.h5')

# Load cancer types from file
with open('../dataset/diseases_names.txt', 'r') as f:
    cancer_types = [line.strip() for line in f.readlines()]

# Perform predictions
output_file = "../results/predicted_cancer_types.txt"
with open(output_file, "w") as file:
    for i, patient_embedding in enumerate(patient_embeddings):
        # Concatenate patient and default cancer embeddings
        combined_embedding = np.concatenate([patient_embedding, default_cancer_embedding])
        combined_embedding = np.expand_dims(combined_embedding, axis=0)

        # Model prediction
        prediction = np.argmax(model.predict(combined_embedding, verbose=0))

        # Map index back to cancer type
        cancer_type = cancer_types[prediction]

        file.write(f"Patient {i + 1}: Predicted Cancer Type -> {cancer_type}\n")

print(f"Predictions saved in '{output_file}'")
