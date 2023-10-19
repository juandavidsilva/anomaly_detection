import torch
import numpy as np
import yaml
from pathlib import Path
from torch import nn
import json
import logging
import math
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT   = Path.cwd()
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
            ("model_VB", "VB_th", "voltage_battery"),
            ("model_CB", "CB_th", "current_battery"),
            ("model_VS", "VS_th", "voltage_solar"),
            ("model_CS", "CS_th", "current_solar"),
        ]

        for model_attr, threshold_attr, config_key in models_config:
            model_path = self.config['models'].get(config_key)
            threshold = self.config['thresholds'].get(config_key)

            if model_path:
                try:
                  model = torch.load(ROOT/model_path)
                except Exception as e:
                  logging.error("Error setup_model :", exc_info=True)
                model = model.to(self.device)
                setattr(self, model_attr, model)
                setattr(self, threshold_attr, threshold)
            else:
                print(f"No model path found for {config_key} in configuration file.")

    def __call__(self,inputdata,model):
        #inputdata= self.single_shoot(inputdata)
        x = 1000
        inputdata=[value if not math.isnan(value) else x for value in inputdata]
        inputdata= self.create_dataset(inputdata)
      
        __, loss = self.predict(getattr(self,model),inputdata)#algo esta mal con la creacion del dataset
        
        if loss[0] >= self.VB_th:
           ans="Anomaly"
        else:
           ans="Normal"
        return [ans,loss[0],self.VB_th]       
        
    def single_shoot(self,input):
        np_sequence        = np.asarray(input, dtype=np.float32)
        tensor_sequence    = torch.tensor(np_sequence).unsqueeze(1).float() 
        seq_len,n_features = tensor_sequence.shape[0],tensor_sequence.shape[1]
        return [tensor_sequence], seq_len, n_features
    
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

if __name__ == "__main__":
    input_list_str = sys.argv[1]
    data = list(map(float, input_list_str.split()))
    detector = AnomalyDetector()
    result   = detector(data,'model_VB')

    print(json.dumps(result))




