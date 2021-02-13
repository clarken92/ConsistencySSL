#!/bin/bash

USER="dkie"
PYTHON_EXE="/home/$USER/InstalledSoftwares/anaconda3/envs/Tensorflow/bin/python"
MY_PROJECT="/home/$USER/Working/Workspace/Github/ConsistencySSL"

export PYTHONPATH="$MY_PROJECT:$PYTHONPATH"
$PYTHON_EXE ./run_cifar10.py