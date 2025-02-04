
with open('../dataset/diseases_names.txt', 'r') as diseases_file:
    cancer_types = [line.strip() for line in diseases_file.readlines()]


cancer_type_to_index = {cancer_type: index + 1 for index, cancer_type in enumerate(cancer_types)}


patient_cancer_mapping = {}
with open('../dataset/tcga_drug_responses.txt', 'r') as infile:

    next(infile)
    for line in infile:
        columns = line.strip().split('\t')
        patient_id = columns[0]
        cancer_type = columns[1]
        
        patient_cancer_mapping[patient_id] = cancer_type

with open('../results/patient_cancer_info.txt', 'w') as outfile:
    outfile.write("patient_id\tcancer_type\tcancer_index\n")
    
    for patient_id, cancer_type in patient_cancer_mapping.items():
        cancer_index = cancer_type_to_index.get(cancer_type, "Unknown")  
        outfile.write(f"{patient_id}\t{cancer_type}\t{cancer_index-1}\n")
