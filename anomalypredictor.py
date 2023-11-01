import torch
import yaml
import json 
import numpy  as np

from   typing                                                 import Any
from   pathlib                                                import Path
from   distutils.util                                         import strtobool
from   pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from   tools                                                  import replace_nan_with_mean,check_load_json,json_to_pandas

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AnomalyPredictor():
    def __init__(self, config_path=None):
        self.device = DEVICE
        self.config = self.load_config(config_path or Path(__file__).parent / 'config' / 'config.yaml')
        
        
        if self.config is not None:
            self.setup_models()

    def load_config(self, file_path):
        try:
            with open(file_path, 'r') as stream:
                return yaml.safe_load(stream)
        except FileNotFoundError:
            print(f"Error: {file_path} does not exist.")
        except yaml.YAMLError as exc:
            print(f"Error in configuration file: {exc}")
        return None
    
    def load_input(self, file_path):
        try:
            check,my_input=check_load_json(file_path)
            if not check:
                raise Exception('Error with jsonfile')
            
        except Exception as e: 
            print(f"Error: Check input json: {str(e)}")
        return my_input

    def setup_models(self):
        
        models_config = [
            ("voltage_battery_predictor"),
        ]

        for config_key in models_config:
            model_attr = config_key
            model_path = self.config['models'].get(config_key)

            if model_path:
               model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path=Path(__file__).parent / model_path,map_location=torch.device('cpu'))
               
               setattr(self, model_attr, model)
            else:
                print(f"No model path found for {config_key} in configuration file.")
    
    def create_dataset(self,myinput):
        myinput     = json_to_pandas(myinput)
        for col in myinput.drop(columns=['ID','time_idx']).columns.tolist():
            myinput[col] = replace_nan_with_mean(myinput[col])
        return myinput
    
    def inference(self,my_input):
        raw_predictions = getattr(self,'voltage_battery_predictor').predict(my_input, mode="raw", return_x=True)#todo: check data format in jupyter
        my_output       = raw_predictions.output[-1][:,:,3].detach().cpu().numpy() #7 quantiles by default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98] ,50% is 3 
        my_output
    
    def write_results(self,input_dict,folder_name="output", file_name="output.json"):
        # Convert NumPy arrays to lists
        converted_dict = {key: value.tolist() if isinstance(value, np.ndarray) else value
                            for key, value in input_dict.items()}

        # Define the output folder path
        output_folder = Path(folder_name)
        # Create the output folder if it does not exist
        output_folder.mkdir(parents=True, exist_ok=True)

        # Define the full path for the output file
        file_path = output_folder / file_name

        # Write the converted dictionary to a JSON file
        with file_path.open('w') as json_file:
            json.dump(converted_dict, json_file, indent=4)
        
        print(f"JSON file saved at: {file_path}")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        myoutput  = dict()
        my_input  = self.load_input(Path(__file__).parent / 'input' / 'request.json')
        data      = self.create_dataset(my_input)
        myoutput['voltage_battery_prediction'] = self.inference(data)
        self.write_results(myoutput)
    
if __name__ == "__main__":
    predict = AnomalyPredictor()
    predict()
    

