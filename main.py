from anomalypredictor import AnomalyPredictor
from anomalydetector  import AnomalyDetector,RecurrentAutoencoder,Decoder,Encoder

if __name__ == "__main__":

    detector = AnomalyDetector()
    predict  = AnomalyPredictor()
    detector(anomaly_in_predictions=False)
    predict()
    detector(anomaly_in_predictions=True)
