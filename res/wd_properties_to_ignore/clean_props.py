import json
from tqdm import tqdm
import os

# import properties

# check if label or any of the alias ends with " ID"

additional_props_to_ignore = [
    "P10", # video
    "P18", # image
    "P51", # audio
    "P989", # spoken text audio
    "P2717", # montage image
]

# Load JSON data from the input file
data = None

# this file folder
this_file_path = os.path.dirname(os.path.realpath(__file__))
properties_path = os.path.join(this_file_path, "all_wd_properties.json")

file = properties_path
with open(file, "r") as f:
    data = json.load(f)

# List to hold IDs to remove
ids_to_remove = []

# Iterate over each JSON element
for item in tqdm(data):
    labels = [item.get("label", "")] if item.get("label") is not None else []
    alias = item.get("alias", []) if item.get("alias") is not None else []
    
    # Check for " ID" in labels and alias
    has_id = any(label.endswith((" ID", 'URL')) for label in labels) or \
             any(al.endswith((" ID", 'URL')) for al in alias if len(alias) > 0)
    
    if has_id:
        ids_to_remove.append({"id": item.get("id"), "label": labels, "alias": alias})

# Write the IDs to a JSON file
output_file = os.path.join(this_file_path, "ids_to_remove_V2.json")
with open(output_file, "w") as f:
    json.dump(ids_to_remove, f, indent=4)

print(f"IDs written to {output_file}: {ids_to_remove}")