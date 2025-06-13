# LIBLINEAR-Weights-Kernelized

## tl;dr

-   Support vector machine (SVM) solvers [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) and [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) have two differences: one is the use of kernel functions (only LIBSVM accepts), the other is the intercept in the model parameters (only LIBSVM uses).
-   This library is made by modifying LIBLINEAR to train SVM with kernel functions and without intercept.

## TODO

-   The strategy of caching kernel matrix is not well optimized. We will alter this in the future.

## Overview

For the support vector machine (SVM) and similar machine learning models, [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) and [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/) are well-known solvers. However, the author needs slightly different formulations from these two libraries.

The requirements of SVM for the author's work are as follows:

1. specification of weights on training instances (that weight the loss function values),
2. use of kernel functions, and
3. the intercept term is not introduced, or the intercept term is regularized (by L2-regularization)

Here, LIBSVM is fine with points 1 and 2, but not 3. On the other hand, LIBLINEAR is fine with points 1 and 3, but not 2. So the author modified LIBLINEAR so that it accepts kernel functions. More specifically, we introduced a part of LIBSVM implementations into LIBLINEAR codes.

### Caveats

-   Currently, only these models accept kernel functions. An error will be raised if a kernel function is specified for other models.
    -   L2 regularization + hinge loss (Command line option "-t 3"; ordinary SVM)
    -   L2 regularization + squared hinge loss (Command line option "-t 1")
    -   L2 regularization + logistic loss (Command line option "-t 7")
-   Ordinary LIBSVM/LIBLINEAR implementations do not accept instance weights. The ones that accept instance weights are distributed here: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/#weights_for_data_instances .

## How to run

### Prerequisites

The following software components are required:

-   make
-   gcc, g++
-   python (If you need Python wrapper)

If you try Python sample codes, the following Python libraries are required:

-   numpy
-   scipy
-   cvxpy
-   matplotlib

For example of using Anaconda and you would like to create a new environment to run this, the following command will do this:

    conda create -n liblinearWK -c conda-forge pip numpy scipy cvxpy matplotlib
    conda activate liblinearWK
    # Since this library is installed by pip, it is recommended to install pip by conda
    # so that this library is installed only in this environment

### 1. Build C++ library

Run `make` command in the top folder of this library files. Please confirm that it did not end with an error.

If you do not use Python library, the preparation is completed with this. Executable files `train` and `predict` will be produced, so please see command line options by running them.

### 2. Build Python library

Run `make` command in the `python` folder of this library files. Please confirm that it did not end with an error.

### 3. Install Python library

Run `pip install -e .` command in the `python` folder of this library files. It is recommended to create a separate environment for pip and then install it.

If the command did not end with an error, then confirm the installation by running the Python code `from liblinear import liblinear, liblinearutil, commonutil`. (It is easy to try in Python interactive mode.)

### 4. Run the sample code

Move to `example` folder of this library files, and run the following command:

    python comparison_CVXPY_LIBLINEAR.py ../splice_scale 0.1 2 1

This will train SVM for the data file "../splice_scale" with the regularization strength 0.1, weights for positive instances 2 and for negative instances 1. Then we compare it with the implementation by [CVXPY](https://www.cvxpy.org/) (optimizer for a large class of convex functions) the trained model parameters in the computation time and the model parameters.

Running this, it will plot the model parameters trained by this implementation and CVXPY, in the ascending order of the ones by this implementation. We will find that they are almost the same.

The computation times in the author's computer was as follows.

|Kernel|Optimizer for:|Training by CVXPY (s)|Training by This implementation (s)|
|:----:|:--------------:|------:|-----:|
|Linear|Primal variables| 1.1540|0.1979|
|Linear|Dual variables  | 1.5910|1.4058|
|RBF   |Dual variables  |12.4578|0.0641|

(Note that the optimizer for primal variables can be used only for linear kernel.)
