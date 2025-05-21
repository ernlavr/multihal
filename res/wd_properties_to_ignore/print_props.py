import json

file_name = 'res/wd_properties_to_ignore/numerical_props.json'

tmp = []
with open(file_name, 'r') as f:
    data = json.load(f)
    for i in data['data']:
        tmp.append(i['property'].split('/')[-1])
        
assert len(tmp) == len(list(set(tmp))), "There are duplicates in the file"

print(f"Total number of properties: {len(tmp)}")
print(f"Properties: {tmp}")