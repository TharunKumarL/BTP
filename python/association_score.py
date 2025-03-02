import h5py
import numpy as np
import os
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to explore and load H5 file
def explore_h5_file(file_path):
    data = []
    with h5py.File(file_path, 'r') as file:
        file.visititems(lambda name, obj: data.extend(obj[:]) if isinstance(obj, h5py.Dataset) else None)
    return np.array(data)

# Define file paths
data_dir = "../preprocesseddata"
results_dir = "../results"

# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)

# Load embeddings
disease_embeddings = explore_h5_file(os.path.join(data_dir, "G_diseases.h5py"))  # (21, 21)
gene_embeddings = explore_h5_file(os.path.join(data_dir, "G_genes.h5py"))  # (15224, 70)

# Expand disease_embeddings to match the number of genes
expanded_disease_embeddings = np.tile(disease_embeddings.mean(axis=0), (gene_embeddings.shape[0], 1))  # (15224, 21)

# Concatenate embeddings
X = np.concatenate((expanded_disease_embeddings, gene_embeddings), axis=1)  # (15224, 91)

# Normalize features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Load or generate labels
# Ideally, load ground-truth labels from a file (e.g., 'cancer_gene_labels.txt')
# If not available, generate pseudo-labels using cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

y = []
for i, gene in enumerate(X):
    scores = [cosine_similarity(gene, disease) for disease in disease_embeddings]
    best_cancer_type = np.argmax(scores)  # Get most similar cancer type
    y.append(best_cancer_type)

y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Save predictions
output_file = os.path.join(results_dir, "cancer_gene_predictions.txt")
with open(output_file, "w") as f:
    for i, pred in enumerate(y_pred):
        f.write(f"Gene_{i+1}, Predicted_Cancer_Type_{pred}\n")

print(f"Predictions saved to {output_file}")
