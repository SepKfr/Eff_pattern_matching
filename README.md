# An Adaptive Temporal Attention Mechanism to Address Distribution Shifts

This repository contains the code and repoducibility instructions of our ```Adaptive Temporal Attention Mechanism to Address Distribution Shifts``` paper accepted at NeurIPS Robustness for sequence modeling workshop.  

## Abstract

With the goal of studying robust sequence modeling via time series, we propose a robust multi-horizon forecasting approach that adaptively reacts to distribution shifts on relevant time scales. It is common in many forecasting domains to observe slow or fast forecasting signals at different times. For example wind and river forecasts are slow changing during drought, but fast during storms. Our approach is based on the transformer architecture, that across many domains, has demonstrated significant improvements over other architectures. Several works benefit from integrating a temporal context to enhance the attention mechanism's understanding of the underlying temporal behavior. In this work, we propose an adaptive temporal attention mechanism that is capable to dynamically adapt the temporal observation window as needed. Our experiments on several real-world datasets demonstrate significant performance improvements over existing state-of-the-art methodologies

Here you can find the link to our paper: https://openreview.net/pdf?id=5aIwxkn0MzC

## Requirements

```
python >= 3.10.4
torch >= 1.13.0
optuna >= 3.0.4
gpytorch >= 1.9.0
numpy >= 1.23.5
```

## How to run:

```
Command line arguments:

exp_name: str    the name of the dataset
name:str   name of the end-to-end forecasting model (for saving model purpose)
attn_type:str    the type of the attention model (ATA, autofomer, informer, conv_attn)
seed:int         random seed value
cuda:str         which GPU
```


# one example with traffic dataset and ATA forecasting model
```
python train.py --exp_name solar --model_name ATA_gp --attn_type ATA --seed 4293 --cuda cuda:0
```
