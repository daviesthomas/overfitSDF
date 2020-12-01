# overfitSDF

Implementation of the architecture presented in "Overfit Neural Networks as a Compact Shape Representation". 

## Install

Our main two dependencies are on tensorflow for neural network implementation and libigl for all geometry processing related tasks. 

We package all of our dependencies in the included environment.yaml file which can be installed using conda. 

```
conda env create -f environment.yml
```

Once created the environment can be activated with
```
conda activate overfitSDF
```

From now on we assume all commands are ran from within this conda environment. 

## Usage

### Convert Mesh to Weight-encoded neural implicit

```
python train.py data/bumpy-cube.obj
```

*NOTE*: cube marching is not currently implemented, this repo is a work in progress as we port code from libigl C++ to libigl-python-bindings for easier usage. We apologize for the delay! 
