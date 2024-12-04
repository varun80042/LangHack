import json
import re

def clean_value(value):
    """
    Clean the value by removing unwanted characters, leading/trailing whitespace, and newlines.
    """
    if value:
        # Remove unwanted characters like newlines and non-breaking spaces
        value = re.sub(r'[\t\n\u00a0]+', ' ', value)  # Replace newlines and non-breaking spaces with a space
        return value.strip()  # Remove leading/trailing whitespace
    return value

def clean_data(data):
    """
    Clean the data by iterating through each key-value pair and cleaning the value.
    """
    cleaned_data = {}
    for key, value in data.items():
        cleaned_data[key] = clean_value(value)
    return cleaned_data

def read_and_clean_json(file_path):
    """
    Read a JSON file, clean the data for each object, and return a list of cleaned objects.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    cleaned_data = []
    for obj in data:
        cleaned_obj = clean_data(obj)
        cleaned_data.append(cleaned_obj)

    return cleaned_data

def write_cleaned_json(file_path, cleaned_data):
    """
    Write the cleaned data to a new JSON file.
    """
    with open(file_path, 'w') as file:
        json.dump(cleaned_data, file, indent=2)

# Example usage
input_file_path = 'E:\\miscE\\ml\\LLM_Hackathon\\combined_dataset.json'
output_file_path = 'E:\\miscE\\ml\\LLM_Hackathon\\cleaned_combined_dataset.json'  # Changed output file name to avoid overwriting
cleaned_data = read_and_clean_json(input_file_path)
write_cleaned_json(output_file_path, cleaned_data)

print(f"Cleaned data written to: {output_file_path}")
