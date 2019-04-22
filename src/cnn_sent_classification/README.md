## Convolutional Neural Networks and Recurrent Neural Networks for Toxic Comment Classification

This repo contains implementations of some of the papers and experiments on how to use different CNN and RNN models for toxic comment classification.

### Usage
- Run the `data/download.sh` script to download data from kaggle, but before please make sure to install `kaggle-api` using `pip install kaggle-api`
- Run `main.py` to run a particular model defined in the `models/` folder.
- I have also added model with best performance on the holdout set in the `results/` folder, you can use that model by passing `-infer` flag while running the `main.py` script.


### Installation

```
virtualenv env -p python3.7
source env/bin/activate
pip install -r requirements.txt
```

### Run the models
- You can run the model in `training` and `inference` mode, once you have trained different models and saved them to `results` folder, just run main.py again without the `infer` flag.
