import os
import json

# Directories
input_dir = "E:\miscE\ml\LLM_Hackathon\datasets\datasets\microlabs_usa"  # Replace with the folder containing your JSON files
output_file = "combined_dataset.json"

# Standardized fields
required_fields = [
    "product_name", "INGREDIENTS AND APPEARANCE", "dosage", 
    "contraindications", "side_effects", "warnings"
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
