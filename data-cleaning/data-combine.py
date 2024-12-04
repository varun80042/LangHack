import os
import json

# Directories
input_dir = "D:\work\sem 7\Large Language Models\LLM_Hackathon\datasets\microlabs_usa"  # Replace with the folder containing your JSON files
output_file = "combined_dataset.json"

# Standardized fields
required_fields = [
    "product_name", "INGREDIENTS AND APPEARANCE", "PACKAGE LABEL.PRINCIPAL DISPLAY PANEL", 
    "HOW SUPPLIED:", "DOSAGE AND ADMINISTRATION:", "OVERDOSAGE:", "ADVERSE REACTIONS:", "PRECAUTIONS:", "WARNINGS:", "CONTRAINDICATIONS:", "INDICATIONS AND USAGE:", "CLINICAL PHARMACOLOGY:", "DESCRIPTION:",
    "HIGHLIGHTS OF PRESCRIBING INFORMATION", "Table of Contents", "1 INDICATIONS AND USAGE", "2 DOSAGE AND ADMINISTRATION", "3 DOSAGE FORMS AND STRENGTHS", "4 CONTRAINDICATIONS", "5 WARNINGS AND PRECAUTIONS", "6 ADVERSE REACTIONS", "7 DRUG INTERACTIONS", "8 USE IN SPECIFIC POPULATIONS", "10 OVERDOSAGE", "11 DESCRIPTION", "12 CLINICAL PHARMACOLOGY", "13 NONCLINICAL TOXICOLOGY", "14 CLINICAL STUDIES", "15 REFERENCES", "16 HOW SUPPLIED/STORAGE AND HANDLING", "17 PATIENT COUNSELING INFORMATION", "" 
]

def clean_json(file):
    with open(file, "r") as f:
        data = json.load(f)
    # Ensure all required fields are present
    return {field: data.get(field, None) for field in required_fields}

def combine_json_files(input_dir, output_file):
    dataset = []
    for file in os.listdir(input_dir):
        if file.endswith(".json"):
            dataset.append(clean_json(os.path.join(input_dir, file)))
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=4)
    print(f"Dataset combined and saved to {output_file}")

combine_json_files(input_dir, output_file)
