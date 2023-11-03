import torch
import yaml
import numpy as np

import logging
import math
from   pathlib        import Path
from   torch          import nn
from   distutils.util import strtobool
from   typing         import Any
from   tools          import replace_nan_with_mean,json_to_pandas,write_results,check_load_json

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

    def load_input(self, file_path,predictionjson=False):
        try:
            check,my_input=check_load_json(file_path,predictionjson=predictionjson)
            if not check:
                raise Exception('Error with jsonfile')
            
        except Exception as e: 
            print(f"Error: Check input json: {str(e)}")
        return my_input
    
    def setup_models(self):
        
        models_config = [
            ("Voltage-Battery", "VB_th", "Voltage-Battery"),
            ("Current-Battery", "CB_th", "Current-Battery"),
            ("Voltage-Solar", "VS_th", "Voltage-Solar"),
            ("Current-Solar", "CS_th", "Current-Solar"),
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

    def __call__(self, *args: Any, **kwds: Any) -> Any:
      myoutput  = dict()
      if kwds['anomaly_in_predictions']:
        v = 'Voltage-Battery'
        inputdata=self.anomaly_in_predictions()
        
        sequences = torch.tensor(inputdata['Voltage-battery-predictor'])
        sequences = [torch.tensor([item]) for item in sequences]
        sequences = torch.stack(sequences)
        dataset   = sequences

        n_seq, seq_len, n_features = torch.stack([dataset]).shape

        __, loss = self.predict(getattr(self,v),[dataset, seq_len, n_features])
        
        if loss[0] >= self.config['thresholds'][v]:
            ans="Anomaly"
        else:
            ans="Normal "
        myoutput[v]= [ans,loss[0],self.sigmoid_threshold(loss[0],self.config['thresholds'][v])]
        write_results({'anomaly_in_predictions':myoutput},folder_name=self.config['data']['output'],overwrite=False)
      else:

        inputdata = self.load_input(Path(__file__).parent /  self.config['data']['input'] / 'request.json')
        inputdata= self.create_dataset(inputdata)
        for v in ['Voltage-Battery','Current-Battery','Voltage-Solar','Current-Solar']:
          sequences = torch.tensor(inputdata[v])
          sequences = [torch.tensor([item]) for item in sequences]
          sequences = torch.stack(sequences)
          dataset   = sequences

          n_seq, seq_len, n_features = torch.stack([dataset]).shape

          __, loss = self.predict(getattr(self,v),[dataset, seq_len, n_features])
          
          if loss[0] >= self.config['thresholds'][v]:
              ans="Anomaly"
          else:
              ans="Normal "
          myoutput[v]= [ans,loss[0],self.sigmoid_threshold(loss[0],self.config['thresholds'][v])]
        write_results({'anomaly_detector':myoutput},folder_name=self.config['data']['output'])
           
    def create_dataset(self,myinput,predictionjson=False):
        x = 1000 #<--penalty if batch of NaN is present, bias param.
        myinput   = json_to_pandas(myinput,doubled=False,predictionjson=predictionjson)
        for col in myinput.drop(columns=['ID','time_idx']).columns.tolist():
            myinput[col] = replace_nan_with_mean(myinput[col])
            myinput[col]   = [value if not math.isnan(value) else x for value in myinput[col]]

        return myinput

    def predict(self,model, dataset):

        predictions, losses = [], []

        criterion = nn.L1Loss(reduction='sum').to(self.device)

        with torch.no_grad():

            model = model.eval()

            seq_true = dataset[0] 

            seq_true = seq_true.to(self.device)

            seq_pred = model(seq_true.to(dtype=torch.float32))

            loss = criterion(seq_pred, seq_true)

            predictions.append(seq_pred.cpu().numpy().flatten())

            losses.append(loss.item())

        return predictions, losses

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

    def anomaly_in_predictions(self):
      inputdata = self.load_input(Path(__file__).parent /  self.config['data']['output'] / 'output.json',predictionjson=True)
      inputdata= self.create_dataset(inputdata,predictionjson=True)
      return inputdata
    

if __name__ == "__main__":

      detector = AnomalyDetector()
      #detector()
      detector(anomaly_in_predictions=True)
