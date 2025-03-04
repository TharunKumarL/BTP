import numpy as np
import h5py
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def explore_h5_file(file_path):
    data = []
    with h5py.File(file_path, 'r') as file:
        file.visititems(lambda name, obj: data.extend(obj[:]) if isinstance(obj, h5py.Dataset) else None)
    return np.array(data)

# Load embeddings
gene_embeddings = explore_h5_file('../preprocesseddata/G_genes.h5py')
disease_embeddings = explore_h5_file('../preprocesseddata/G_diseases.h5py')

# Ensure both embeddings have the same size
max_dim = max(gene_embeddings.shape[1], disease_embeddings.shape[1])

def pad_embeddings(embeddings, target_dim):
    pad_width = target_dim - embeddings.shape[1]
    return np.pad(embeddings, ((0, 0), (0, pad_width)), mode='constant') if pad_width > 0 else embeddings

gene_embeddings = pad_embeddings(gene_embeddings, max_dim)
disease_embeddings = pad_embeddings(disease_embeddings, max_dim)

# Generate training data by concatenating gene and disease embeddings
X = np.array([np.hstack((gene_embeddings[i], disease_embeddings[j])) 
              for i in range(len(gene_embeddings)) for j in range(len(disease_embeddings))])

# Generate synthetic labels (replace with real scores if available)
y = np.random.rand(X.shape[0])  

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Predict on the entire dataset (all gene-disease pairs)
interaction_scores = model.predict(X).reshape(len(gene_embeddings), len(disease_embeddings))

# Save scores to a file
np.savetxt("../results/interaction_scores.txt", interaction_scores, fmt='%f')

# Evaluate model performance on test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Print a sample of the computed scores
print("Sample Interaction Scores (Top 5 rows):")
print(interaction_scores[:5])
