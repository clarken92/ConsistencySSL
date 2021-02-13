# Semi-Supervised Learning with Variational Bayesian Inference and Maximum Uncertainty Regularization


This repository contains the official implementation of our paper:
> [**Semi-Supervised Learning with Variational Bayesian Inference and Maximum Uncertainty Regularization**](https://arxiv.org/abs/2012.01793)
>
> [Kien Do](https://twitter.com/kien_do_92), [Truyen Tran](https://twitter.com/truyenoz), Svetha Venkatesh

__Accepted at AAAI 2021.__


## Contents
1. [Requirements](#requirements)
1. [Features](#features)
0. [Repository structure](#repository-structure)
0. [Setup](#setup)
0. [Downloading and preprocessing data](#downloading-and-preprocessing-data)
0. [Training](#training)
0. [Citation](#citation)

## Requirements
Tensorflow >= 1.8

The code hasn't been tested with Tensorflow 2.

This repository is designed to be self-contained. If during running the code, some packages are required, these packages can be downloaded via pip or conda.
Please email me if you find any problems related to this.

## Features
- Support model saving
- Support logging
- Support tensorboard visualization

## Repository structure
Our code is organized in 5 main parts:
- `models`: Containing models used in our paper, including Pi, MT, MT+VD, MT+MUR,....
- `components`: Containing implementation for the CNN13 classifier.
- `my_utils`: Containing utility functions.
- `data_preparation`: Containing code for downloading and preprocessing datasets.
- `working`: Containing scripts for training models.

**IMPORTANT NOTE**: Since this repository is organized as a Python project, I strongly encourage you to import it as a project to an IDE (e.g., PyCharm). By doing so, the path to the root folder of this project will be automatically added to PYTHONPATH when you run the code via your IDE. Otherwise, you have to explicitly add it when you run in terminal. Please check `run_cifar10.sh` (or `run_cifar100.sh`, `run_svhn.sh`) to see how it works.

## Setup
The setup for training is **very simple**. All you need to do is opening the `global_settings.py` file and changing the values of the global variables to match yours. The meanings of the global variables are given below:
* `PYTHON_EXE`: Path to your python interpreter.
* `PROJECT_NAME`: Name of the project, which I set to be `'ConsistencySSL'`.
* `PROJECT_DIR`: Path to the root folder containing the code of this project.
* `RESULTS_DIR`: Path to the root folder that will be used to store results for this project.  
* `RAW_DATA_DIR`: Path to the root folder that contains raw datasets. By default, the root directory of the CIFAR10/CIFAR100/SVHN dataest is `$RAW_DATA_DIR/ComputerVision/[dataset_name]`.

## Downloading and preprocessing data
Before training, you need to download and preprocess datasets. Scripts for each dataset are provided in `data_preparation/[dataset name]`. You simply need to run them in order.

For example, to prepare the CIFAR10 dataset, run the following commands:
```shell
export PYTHONPATH="[path to this project]:$PYTHONPATH"
python 1_process_data.py
python 2_generate_zca.py
```

## Training
Once you have setup everything in `global_settings.py`, you can start training by running the following commands in your terminal:
```shell
export PYTHONPATH="[path to this project]:$PYTHONPATH"
python train.py [required arguments]
```
**IMPORTANT NOTE**: If you run using the commands above, please remember to provide all **required** arguments specified in `train.py` otherwise errors will be raised.

However, if you are too lazy to type arguments in the terminal (like me :sweat_smile:), you can set these arguments in the `run_config` dictionary in `run_cifar10.py` (or `run_cifar100.py`, `run_svhn.py`) and simply run this file:
```shell
export PYTHONPATH="[path to this project]:$PYTHONPATH"
python run_cifar10.py
```

I also provide a `run_cifar10.sh` file as an example for you.

## Citation
If you find this repository useful for your research, please consider citing our paper:

```bibtex
@article{do2020semi,
  author  = {Do, Kien and Tran, Truyen and Venkatesh, Svetha},
  title   = {Semi-Supervised Learning with Variational Bayesian Inference and Maximum Uncertainty Regularization},
  journal = {arXiv preprint arXiv:2012.01793},
  year    = {2020},
}
```
