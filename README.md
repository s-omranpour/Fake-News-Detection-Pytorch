#  Fake-News-Detection-Pytorch

This project is a pytorch implementation of the paper [CSI: A Hybrid Deep Model for Fake News Detection"](https://dl.acm.org/citation.cfm?id=3132877).

Several modifications have been made on the [original code](https://github.com/sungyongs/CSI-Code) including:

- minor bugs were fixed
- code is cleaner, standard, organized, and more understandable
- the main model is reimplemented using Pytorch and Pytorch_lightning
- training is integrated with Ray Tune for the sake of hyperparameter tuning
    
    
## Getting started
1. First, use the [data preprocessing notebook](data%20preprocessing.ipynb) to prepare your data for the model. You should apply the code for different splits separately.

2. Then, use the [train notebook](train.ipynb) to train your model and see the results.
