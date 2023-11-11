import argparse
from anomalypredictor import AnomalyPredictor
from anomalydetector  import AnomalyDetector,RecurrentAutoencoder,Decoder,Encoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly Detector and Predictor")
    parser.add_argument('input_file', type=str, help="Path to the input file")
    parser.add_argument('output_path', type=str, help="Where will the output file be written")
    parser.add_argument('--config_file',default=None, type=str, help="config.yaml file")
    args = parser.parse_args()

    detector = AnomalyDetector(args.input_file,args.output_path,config_path=args.config_file)
    predict  = AnomalyPredictor(args.input_file,args.output_path)
    result   = detector(anomaly_in_predictions=False)
    predict(anomaly_input=result)
    detector(anomaly_in_predictions=True)
