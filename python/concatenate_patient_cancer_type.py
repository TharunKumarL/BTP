import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout




def explore_h5_file(file_path):
    d=[]
    with h5py.File(file_path, 'r') as file:
        print("Root file keys:", list(file.keys()))

        # Recursive function to print structure
        def visit_all(name, obj):
            nonlocal d
            if isinstance(obj, h5py.Dataset):
                print(f"\nDataset: {name}")
                print("Shape:", obj.shape)
                print("Dtype:", obj.dtype)
                d=obj[:]
                print("Data snippet:", d)  # Display first few elements
                return d
                
            elif isinstance(obj, h5py.Group):
                print(f"\nGroup: {name}")

        file.visititems(visit_all)
        return d

disease_embeddings = explore_h5_file('../preprocesseddata/G_diseases.h5py')
patient_embeddings = explore_h5_file('../preprocesseddata/G_patients.h5py')
print("Disease Embeddings shape:", len(disease_embeddings))

patient_cancer_data = pd.read_csv('../results/patient_cancer_info.txt', sep='\t')

# Load cancer types from diseases_names.txt
with open('../dataset/diseases_names.txt', 'r') as f:
    cancer_types = [line.strip() for line in f.readlines()]

# Create a mapping from cancer type to an index
cancer_type_to_index = {cancer_type: index for index, cancer_type in enumerate(cancer_types)}

print("Cancer Type to Index mapping:", cancer_type_to_index)




X = [] 
y = []  

for index, row in patient_cancer_data.iterrows():
    cancer_type = row['cancer_type']
    patient_id = row['patient_id']
    print(f"Processing patient {patient_id} with cancer type {cancer_type}")
    cancer_index = cancer_type_to_index[cancer_type]
    print(f"Cancer index: {cancer_index}")
    cancer_embedding = disease_embeddings[cancer_index]
    patient_embedding = patient_embeddings[patient_id-1]
    
    combined_embedding = np.concatenate([patient_embedding, cancer_embedding])
    
    X.append(combined_embedding)
    y.append(cancer_index)

X = np.array(X)
y = np.array(y)

y = LabelEncoder().fit_transform(y) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(cancer_type_to_index), activation='softmax')) 

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

model.save('../models/patient_cancer_model.h5')
