import pandas as pd
import json

# Load the CSV file
solchan_csv_path = 'solchan.csv'
df = pd.read_csv(solchan_csv_path)

# Convert CSV data to JSON format similar to kiapi.json structure
json_data = {}
for idx, row in df.iterrows():
    # Using 'x' and 'y' similar to the structure of kiapi.json
    # Setting a fixed category "driving" and choosing the 'alpha' column for the final value
    json_data[str(idx)] = [row['x'], row['y'], "driving", row['alpha']]

# Replace all instances of 0.0 with 5
for key, values in json_data.items():
    json_data[key] = [5 if v == 0.0 else v for v in values]

# Save the updated JSON data
updated_json_path = 'solchan.json'
with open(updated_json_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)

