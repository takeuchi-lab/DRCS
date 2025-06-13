# Code for the experiment in the paper "Distributionally Robust Coreset Selection under Covariate Shift"

Authors: Tomonari Tanaka, Hiroyuki Hanada, Hanting Yang, Aoyama Tatsuya, Yu Inatsu, Akahane Satoshi, Yoshito Okura, Noriaki Hashimoto, Taro Murayama, Hanju Lee, Shinya Kojima, Ichiro Takeuchi

[Transactions on Machine Learning Research](https://jmlr.org/tmlr/), to appear

URL: [https://openreview.net/forum?id=Eu7XMLJqsC](https://openreview.net/forum?id=Eu7XMLJqsC)

## Setup

Programs are implemented in Python.

The following packages can be installed via `pip` or `conda`:

-   `numpy`
-   `scipy`
-   `scikit-learn`
-   `matplotlib`
-   `cvxpy`
-   `pytorch` (required only when we specify `cifar10` as a dataset)
-   `torchvision` (required only when we specify `cifar10` as a dataset)

The following package can be installed by compiling with C++ compiler (`gcc`, `g++` and `make` are required) and then via `pip`:

- `liblinear-weighted-kernelized` (a modification of [https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/#weights_for_data_instances](LIBLINEAR with weights for data instances) for kernel models); see `source` folder and the file `README.md` in it for installation.

## Experiments

Run the followings:

- `./experiment_drcs.sh`: Model performance of DRCS

## Copyright notice

All programs are written by the authors.

Dataset files are retrieved from "LIBSVM Data" by Rong-En Fan (National Taiwan University).
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
