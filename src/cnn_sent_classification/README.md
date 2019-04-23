## Convolutional Neural Networks and Recurrent Neural Networks for [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

This repo contains implementations of some of the papers and experiments on how to use different CNN architectures for classification task.

### Usage
- Clone the rep0
- `cd src/cnn_sent_classificaiton/`
- Run the `data/download.sh` script to download data from kaggle, but before please make sure to install `kaggle-api` using `pip install kaggle-api` from [Kaggle API](https://github.com/Kaggle/kaggle-api)
- Run `sh train_full.sh` to train model defined in the `models/` folder.
- Model with best performance on the holdout set would get stored in the `results/` folder, you can use that model by passing `-infer True -load True` flag while running the `sh train_full.sh` script.


### Installation

```
virtualenv env -p python3.7
source env/bin/activate
pip install -r requirements.txt
```

### Resources
-[Convolutional Neural Networks for Sentence Classification Yoon Kim, 2014](https://arxiv.org/abs/1408.5882)