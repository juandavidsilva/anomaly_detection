import numpy as np
import json 
import pandas as pd
from pathlib import Path

def replace_nan_with_mean(arr):
    arr = np.asanyarray(arr)

    valid_replace = np.concatenate(([False], np.isnan(arr[1:-1]) & ~np.isnan(arr[:-2]) & ~np.isnan(arr[2:]), [False]))
    
    arr[valid_replace] = (arr[np.roll(valid_replace, -1)] + arr[np.roll(valid_replace, 1)]) / 2
    
    return arr.tolist()
  
def check_load_json(file_path,predictionjson=False):
    # Load the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    if predictionjson:
        if len(data['predictor']['Voltage-battery-predictor']) != 24:
            print(f"The key '{key}' does not have a list of 24 items.")
            return [False,data]
    else:
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

def json_to_pandas(json_input,doubled=True,predictionjson=False):
    # Load json data
    if predictionjson:
        data = json_input['predictor']
    else:
        data = json_input['request']
    df = pd.DataFrame(data)
    df=refactor_df(df,doubled=doubled)
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

def refactor_df(df,doubled=True,redo_id=True):
    zeros_df               = df
    if doubled:
        doubled_df         = pd.concat([df, zeros_df], ignore_index=True)
    else:
        doubled_df         = df
    # Add 'ID' column with all rows having the string '0'
    if redo_id:
        doubled_df['ID']       = '0'
    # Add 'time_idx' column with consecutive integers starting from 0
    doubled_df['time_idx'] = range(len(doubled_df))

    return doubled_df

def write_results(input_dict, folder_name="output", file_name="output.json", overwrite=False):
    # Convert NumPy arrays to lists
    converted_dict = {key: value.tolist() if isinstance(value, np.ndarray) else value
                      for key, value in input_dict.items()}

    # Define the output folder path
    output_folder = Path(__file__).parent / folder_name

    # Create the output folder if it does not exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Define the full path for the output file
    file_path = output_folder / file_name

    # Check if the file exists
    if file_path.is_file() and not overwrite:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            # Assuming you want to update the JSON with new values from converted_dict
            data.update(converted_dict)
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    else:
        # If the file does not exist or overwrite is True, write the converted dictionary to a new JSON file
        with open(file_path, 'w') as json_file:
            json.dump(converted_dict, json_file, indent=4)

    print(f"JSON file saved at: {file_path}")

def csv_to_pandas(csv_file_path,sep=','):
    #1. Open csv file
    odf = pd.read_csv(csv_file_path,sep=sep)
    df  = odf.copy()
    df  = refactor_df(df,doubled=False,redo_id=False)
    return df

def concat_and_save_to_csv(list_of_dfs, output_csv):
    # Concatenate DataFrames along the vertical axis
    combined_df = pd.concat(list_of_dfs, axis=0)
    
    # Reset index to ensure correct consecutiveness
    combined_df = combined_df.reset_index(drop=True)
    
    # Assign new values to 'time_idx' column
    combined_df['time_idx'] = range(1, len(combined_df) + 1)
    
    # Save the DataFrame to a CSV file
    combined_df.to_csv(Path(output_csv) / 'output.csv', index=False)


if __name__ == "__main__":
    pass