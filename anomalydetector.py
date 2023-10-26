import torch
import numpy as np
import yaml
from pathlib import Path
from torch import nn
import json
import logging
import argparse
import math
from distutils.util import strtobool

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(filename='log.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
class Encoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):

    super(Encoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features

    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(

      input_size=n_features,

      hidden_size=self.hidden_dim,

      num_layers=1,

      batch_first=True

    )

    self.rnn2 = nn.LSTM(

      input_size=self.hidden_dim,

      hidden_size=embedding_dim,

      num_layers=1,

      batch_first=True

    )

  def forward(self, x):

    x = x.reshape((1, self.seq_len, self.n_features))

    x, (_, _) = self.rnn1(x)

    x, (hidden_n, _) = self.rnn2(x)

    return hidden_n.reshape((self.n_features, self.embedding_dim))

class Decoder(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1):

    super(Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim

    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.LSTM(

      input_size=input_dim,

      hidden_size=input_dim,

      num_layers=1,

      batch_first=True

    )

    self.rnn2 = nn.LSTM(

      input_size=input_dim,

      hidden_size=self.hidden_dim,

      num_layers=1,

      batch_first=True

    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):

    x = x.repeat(self.seq_len, self.n_features)

    x = x.reshape((self.n_features, self.seq_len, self.input_dim))

    x, (hidden_n, cell_n) = self.rnn1(x)

    x, (hidden_n, cell_n) = self.rnn2(x)

    x = x.reshape((self.seq_len, self.hidden_dim))

    return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):

    super(RecurrentAutoencoder, self).__init__()

    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(DEVICE)

    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(DEVICE)

  def forward(self, x):

    x = self.encoder(x)

    x = self.decoder(x)

    return x

class AnomalyDetector():
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
            logging.error(f"Error: {file_path} does not exist.",exc_info=True)
            print(f"Error: {file_path} does not exist.")
        except yaml.YAMLError as exc:
            logging.error(f"Error in configuration file: {exc}",exc_info=True)
            print(f"Error in configuration file: {exc}")
        return None

    def setup_models(self):
        
        models_config = [
            ("voltage_battery", "VB_th", "voltage_battery"),
            ("current_battery", "CB_th", "current_battery"),
            ("voltage_solar", "VS_th", "voltage_solar"),
            ("current_solar", "CS_th", "current_solar"),
        ]

        for model_attr, threshold_attr, config_key in models_config:
            model_path = self.config['models'].get(config_key)
            threshold = self.config['thresholds'].get(config_key)

            if model_path:
                try:
                  model = torch.load(Path(__file__).parent / model_path)
                except Exception as e:
                  logging.error("Error setup_model :", exc_info=True)
                
                model = model.to(self.device)
                setattr(self, model_attr, model)
                setattr(self, threshold_attr, threshold)
            else:
                print(f"No model path found for {config_key} in configuration file.")

    def __call__(self,inputdata,model,remove_nan):
        
        if remove_nan:
          inputdata = self.replace_nan_with_mean(inputdata)
        
        x = 1000 #<--penalty if batch of NaN is present, bias param.
        inputdata=[value if not math.isnan(value) else x for value in inputdata]
        inputdata= self.create_dataset(inputdata)
      
        __, loss = self.predict(getattr(self,model),inputdata)
        
        if loss[0] >= self.config['thresholds'][model]:
           
           ans="Anomaly Range"
        else:
           ans="Normal Range "
        return [ans,loss[0],self.config['thresholds'][model],self.sigmoid_threshold(loss[0],self.config['thresholds'][model])]       
        
    def create_dataset(self,input):
        sequences = torch.tensor(input)
        sequences = [torch.tensor([item]) for item in sequences]
        sequences = torch.stack(sequences)
        dataset   = [sequences]

        n_seq, seq_len, n_features = torch.stack(dataset).shape

        return dataset, seq_len, n_features

    def predict(self,model, dataset):

        predictions, losses = [], []

        criterion = nn.L1Loss(reduction='sum').to(self.device)

        with torch.no_grad():

            model = model.eval()

            for seq_true in dataset[0]: #aca recibo un lista 

                seq_true = seq_true.to(self.device)

                seq_pred = model(seq_true)

                loss = criterion(seq_pred, seq_true)

                predictions.append(seq_pred.cpu().numpy().flatten())

                losses.append(loss.item())

        return predictions, losses

    def replace_nan_with_mean(self,arr):
        arr = np.asanyarray(arr)

        valid_replace = np.concatenate(([False], np.isnan(arr[1:-1]) & ~np.isnan(arr[:-2]) & ~np.isnan(arr[2:]), [False]))
        
        arr[valid_replace] = (arr[np.roll(valid_replace, -1)] + arr[np.roll(valid_replace, 1)]) / 2
        
        return arr.tolist()
    def sigmoid_threshold(self,x, threshold, k=1):
        """
        Applies an adjusted sigmoid function to x such that the output is a probability 
        centered around a specific threshold.

        :param x: Input value (e.g., loss value).
        :param threshold: The loss point that will map to a probability of 0.5.
        :param k: Factor to adjust the slope of the sigmoid function.
        :return: Adjusted probability between 0 and 1.
        """
        adjusted_x  = k * (x - threshold)
        probability = 1 / (1 + np.exp(-adjusted_x))
        probability = float(f"{probability:.{5}g}")
        return probability

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly detector")
    parser.add_argument("input_list",type=str, help="This input must contain sequential values within a list of 24 values.")
    parser.add_argument("model_name",type=str, help="Which model would you like to use? Options: 'voltage_battery', 'current_battery', 'voltage_solar', 'current_solar'.")
    parser.add_argument("single_nan_remove", type=str, default=False, help="Replace NaN values with the average of the predecessor and successor values")
    
    args = parser.parse_args()

    if args.input_list and args.model_name:
      try:
        input_list = [float(item) for item in args.input_list.split(',')]
      except ValueError as e:
        print(f"Error: All items in input_list must be numbers separated by commas. Error details: {str(e)}")
      
      #input_list   = list(map(float, args.input_list.split()))
      my_bool      = bool(strtobool(args.single_nan_remove))
      detector     = AnomalyDetector()
      result       = detector(input_list, args.model_name,my_bool)

      print(json.dumps(result))
    else:
      parser.print_help()