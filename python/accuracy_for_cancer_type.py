import pandas as pd

# Step 1: Load actual cancer type data
actual_data = pd.read_csv("../dataset/tcga_drug_responses.txt", sep="\t")  # Assuming tab-separated file
actual_cancer_types = actual_data.groupby("patient_id")["cancer_type"].first().to_dict()  # Map patient_id to cancer_type

# Step 2: Load predicted data from predictions.txt
predicted_cancer_types = {}
with open("../results/predictions.txt", "r") as file:
    for line in file:
        parts = line.strip().split(": ")
        if len(parts) == 2:
            patient_id = int(parts[0].replace("Patient_", ""))
            predicted_cancer_types[patient_id] = int(parts[1].replace("Cancer_Type_", ""))

# Step 3: Define cancer type mapping
cancer_type_mapping = [
    "GBM-US", "BLCA-US", "LUAD-US", "BRCA-US", "CESC-US", "COAD-US", "HNSC-US", "KIRC-US", 
    "KIRP-US", "LAML-US", "LGG-US", "LIHC-US", "LUSC-US", "OV-US", "PAAD-US", "PRAD-US", 
    "READ-US", "SKCM-US", "STAD-US", "THCA-US", "UCEC-US"
]

# Step 4: Convert predicted numeric indices to actual cancer types
predicted_mapped = {patient: cancer_type_mapping[cancer_type_id] for patient, cancer_type_id in predicted_cancer_types.items()}

# Step 5: Filter only known actual values and calculate accuracy
correct = 0
valid_entries = []
for patient, predicted in predicted_mapped.items():
    actual = actual_cancer_types.get(patient)
    if actual:  # Only consider known actual values
        valid_entries.append((patient, actual, predicted))
        if actual == predicted:
            correct += 1

total = len(valid_entries)
accuracy = (correct / total) * 100 if total > 0 else 0

# Step 6: Save only known values to file
output_file = "../results/cancer_predictions_filtered.txt"
with open(output_file, "w") as f:
    f.write("Patient_ID\tActual_Cancer_Type\tPredicted_Cancer_Type\n")
    for patient, actual, predicted in valid_entries:
        f.write(f"{patient}\t{actual}\t{predicted}\n")

print(f"Filtered results saved to {output_file}")
print(f"Accuracy (only for known values): {accuracy:.2f}%")
