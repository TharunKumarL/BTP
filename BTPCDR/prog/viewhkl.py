import hickle as hkl
print("hickle version:", hkl.__version__)
import os
file_path = os.path.join("..", "data", "GDSC", "drug_graph_feat", "1401.hkl")

if os.path.exists(file_path):
    data = hkl.load(file_path)
    print("Contents of 1401.hkl:", data)
else:
    print("File not found:", file_path)