import numpy as np
import h5py
from tensorflow.keras.models import load_model

def explore_h5_file(file_path):
    data = []
    with h5py.File(file_path, 'r') as file:
        file.visititems(lambda name, obj: data.extend(obj[:]) if isinstance(obj, h5py.Dataset) else None)
    return np.array(data)


patient_embeddings = explore_h5_file('../preprocesseddata/G_patients.h5py')
disease_embeddings = explore_h5_file('../preprocesseddata/G_diseases.h5py')


default_cancer_embedding = disease_embeddings[0]

model = load_model('../models/patient_cancer_model.h5')


with open('../dataset/diseases_names.txt', 'r') as f:
    cancer_types = [line.strip() for line in f.readlines()]

output_file = "../results/predicted_cancer_types.txt"
with open(output_file, "w") as file:
    for i, patient_embedding in enumerate(patient_embeddings):

        combined_embedding = np.concatenate([patient_embedding, default_cancer_embedding])
        combined_embedding = np.expand_dims(combined_embedding, axis=0)

        prediction = np.argmax(model.predict(combined_embedding, verbose=0))

        cancer_type = cancer_types[prediction]

        file.write(f"Patient {i + 1}: Predicted Cancer Type -> {cancer_type}\n")

print(f"Predictions saved in '{output_file}'")
