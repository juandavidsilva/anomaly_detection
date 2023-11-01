import numpy as np
import json 
import pandas as pd
from pathlib import Path

def replace_nan_with_mean(arr):
    arr = np.asanyarray(arr)

    valid_replace = np.concatenate(([False], np.isnan(arr[1:-1]) & ~np.isnan(arr[:-2]) & ~np.isnan(arr[2:]), [False]))
    
    arr[valid_replace] = (arr[np.roll(valid_replace, -1)] + arr[np.roll(valid_replace, 1)]) / 2
    
    return arr.tolist()
  
def check_load_json(file_path):
    # Load the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Check for the existence of the 'request' key
    if 'request' not in data:
        print("The key 'request' is missing in the JSON.")
        return False

    # List of required keys within 'request'
    required_keys = [
        'Voltage-Battery',
        'Current-Battery',
        'Voltage-Solar',
        'Current-Solar'
    ]

    # Verify required keys and the length of the lists
    for key in required_keys:
        if key not in data['request']:
            print(f"The key '{key}' is missing in 'request'.")
            return [False,data]

        # Check that the list contains exactly 24 items
        if len(data['request'][key]) != 24:
            print(f"The key '{key}' does not have a list of 24 items.")
            return [False,data]

    # If everything is correct, return True
    return [True,data]

def json_to_pandas(json_input):
    # Load json data
    data = json_input['request']

    # Create DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Add 'ID' column with all rows having the string '0'
    df['ID'] = '0'

    # Add 'time_idx' column with consecutive integers starting from 0
    df['time_idx'] = range(len(df))

    # Reorder columns so 'ID' and 'time_idx' are at the beginning
    column_order = ['ID', 'time_idx'] + [col for col in df if col not in ['ID', 'time_idx']]
    df = df[column_order]

    return df

def csv_to_json(file_path):
    # Read the Excel file into a pandas DataFrame
    df = pd.read_csv(file_path,sep=';').head(24)

    # Extract data from the DataFrame
    voltage_battery = df["Voltage-Battery"].tolist()
    current_battery = df["Current-Battery"].tolist()
    voltage_solar = df["Voltage-Solar"].tolist()
    current_solar = df["Current-Solar"].tolist()

    # Create the desired JSON structure
    data = {
        "request": {
            "Voltage-Battery": voltage_battery,
            "Current-Battery": current_battery,
            "Voltage-Solar": voltage_solar,
            "Current-Solar": current_solar
        }
    }
    output_path = Path(file_path).with_suffix('.json')
    # Save the JSON data to the output file
    with output_path.open('w') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    csv_to_json('C:/Users/juan.david/projects/datasets/battery/all/test_predictions/1h_test1.csv')