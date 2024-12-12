## Uncover and Unlearn: Fully Test-Time Adaptation via Unlearning Nuisance Factors

This repo contains the source code for the implementation of TIRNU in PyTorch. We address the challenging but practical setting of fully test-time adaptation, where there is no access to source training data, and the model is updated only using unlabelled (target) test data. 

We introduce a novel perspective to achieving **T**est-time **I**nvariant **R**epresentation learning through **U**nlearning **N**uisance (TIRNU).

### Usage
#### Get Started
This repo is built with [PyTorch==2.1.2](https://pytorch.org/), Python=3.9, and the modules in ```requirements.txt```. 

### CIFAR Datasets
To download the CIFAR datasets,
```
export DATADIR=/data/cifar
mkdir ${DATADIR} && cd ${DATADIR}
wget -O CIFAR-10-C.tar https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1
tar -xvf CIFAR-10-C.tar
wget -O CIFAR-100-C.tar https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1
tar -xvf CIFAR-100-C.tar
```
### Model Weights
The models weights can be downloaded from [here](https://drive.google.com/drive/folders/1TqdIpPIEhGpMGiO7vvRDXu7jd4exoSLp?usp=sharing) and placed in ```models/source_weights```.

### Experiments on CIFAR
To run TIRNU on CIFAR-10/CIFAR-100, simply run:
```
bash run.sh
```
The hyperparameters, such as batch size, can be changed in the ```run.sh``` script.

### Acknowledgement
The code is inspired by [TTT++](https://github.com/vita-epfl/ttt-plus-plus.git). We thank the authors for their comprehensive codebase.
