# Code for the experiment in the paper "Distributionally Robust Coreset Selection under Covariate Shift"

## Setup

Programs are implemented in Python.

The following packages can be installed via `pip` or `conda`:

-   `numpy`
-   `scipy`
-   `scikit-learn`
-   `matplotlib`
-   `cvxpy`
-   `pytorch` (required only for `main_image_data.sh`)
-   `torchvision` (required only for `main_image_data.sh`)


The following packages can be installed via `pip`:

-   `jax` (please see [documentation](https://jax.readthedocs.io/en/latest/installation.html) for installation; required only for `main_image_data.sh`)
-   `neural-tangents` (required only for `main_image_data.sh`)


The following package can be installed by compiling with C++ compiler (`gcc`, `g++` and `make` are required) and then via `pip`:

- `liblinear-weighted-kernelized` (a modification of [https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/#weights_for_data_instances](LIBLINEAR with weights for data instances) for kernel models); see `source` folder and the file `README.md` in it for installation.


## Experiments

Run the followings:

- `./experiment_drcs.sh`: Model performance of DRCS

## Copyright notice

All programs are written by the authors.

Dataset files are retrieved from "LIBSVM Data" by Rong-En Fan (National Taiwan University).
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
