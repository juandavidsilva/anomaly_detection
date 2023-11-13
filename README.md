
# Détecteur d'anomalies

Cette fonctionnalité est conçue pour indiquer la présence d'anomalies dans le voltage et le courant des batteries et des panneaux solaires sur les équipements PCMS, en prenant comme mesure le voltage et le courant de la batterie ainsi que le voltage et le courant du panneau solaire sur les dernières 24 heures (1 lecture par heure).






## Caractéristiques :

    
- Détection d'anomalies sur les dernières 24 heures
- Prédiction du comportement du voltage pour les 24 heures suivant la lecture
- Détection d'anomalie sur la prédiction.


## Input and Output

L'utilisateur doit disposer d'un fichier .json d'entrée avec la structure suivante (exemple dans input/demo.json) :


```bash
{
    "request": {
        "Voltage-Battery": [
            13.1899996,
            13.0999994,
            13.0499992,
            .
            .
            .
        ],
        "Current-Battery": [
            0.189999998,
            0.170000002,
            0.179999992,
            .
            .
            .

        ],
        "Voltage-Solar": [
            18.6399994,
            2.21000004,
            .
            .
            .
        ],
        "Current-Solar": [
            0,
            0,
            .
            .
            .

        ]
    }
}
```

Le nombre de lignes par clé doit être égal à 24. Le système tolère la présence de valeurs NaN ; dans le cas où elles sont simples ( [valeur, NaN, valeur]), le système fait la moyenne de la valeur précédente et de la suivante. Si plus d'un NaN est présent, cela sera considéré comme une anomalie.
## Run Locally

Clone the project

```bash
  git clone https://link-to-project
```

Go to the project directory

```bash
  cd my-project
```

Build docker image

```bash
  docker build -t anomaly .
```

run image (with default threshold values)

```bash
  docker run -v path/to/input/file:/app/data -v path/to/output/folder:/app/out anomaly /app/data/input.json /app/out
```

run image with modified threshold vaslues
```bash
docker run -v path/to/input/file:/app/data -v path/to/output/folder:/app/out -v path/to/config/folder:/app/cfg anomaly /app/data/test.json /app/out --config_file /app/cfg/cfg.yaml
```

## Authors

- [@juandavid.silva](http://gitlab.ver-mac.com/Juandavid.silva/anomaly-detector)


