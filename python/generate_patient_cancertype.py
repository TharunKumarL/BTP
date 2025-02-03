# Read the cancer types from diseases_names.txt and create a mapping
with open('../dataset/diseases_names.txt', 'r') as diseases_file:
    cancer_types = [line.strip() for line in diseases_file.readlines()]

# Create a dictionary to map cancer types to their numeric index
cancer_type_to_index = {cancer_type: index + 1 for index, cancer_type in enumerate(cancer_types)}

# Create a dictionary to store patient_id and cancer_type mappings from tcga_drug_responses.txt
patient_cancer_mapping = {}
with open('../dataset/tcga_drug_responses.txt', 'r') as infile:
    # Skip the header line
    next(infile)
    
    # Read all lines and map patient_id to cancer_type
    for line in infile:
        columns = line.strip().split('\t')
        patient_id = columns[0]
        cancer_type = columns[1]
        
        # Store the cancer_type for each patient_id
        patient_cancer_mapping[patient_id] = cancer_type

# Open the output file and write the results
with open('../results/patient_cancer_info.txt', 'w') as outfile:
    # Write the header to the output file
    outfile.write("patient_id\tcancer_type\tcancer_index\n")
    
    # Loop through each patient ID in the patient_cancer_mapping
    for patient_id, cancer_type in patient_cancer_mapping.items():
        cancer_index = cancer_type_to_index.get(cancer_type, "Unknown")  # Get the index for the cancer type
        
        # Write the patient information (patient_id, cancer_type, and cancer_index) to the output file
        outfile.write(f"{patient_id}\t{cancer_type}\t{cancer_index-1}\n")
