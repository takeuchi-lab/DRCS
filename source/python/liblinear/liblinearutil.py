#!/usr/bin/env python

import os, sys
from .liblinear import *
from .liblinear import __all__ as liblinear_all
from .commonutil import *
from .commonutil import __all__ as common_all
import ctypes
from ctypes import c_double

try:
    import numpy as np
    import scipy
    from scipy import sparse
except:
    scipy = None

if sys.version_info[0] < 3:
    range = xrange
    from itertools import izip as zip
    _cstr = lambda s: s.encode("utf-8") if isinstance(s,unicode) else str(s)
else:
    _cstr = lambda s: bytes(s, "utf-8")

__all__ = ['load_model', 'save_model', 'train', 'predict'] + liblinear_all + common_all


def load_model(model_file_name):
    """
    load_model(model_file_name) -> model

    Load a LIBLINEAR model from model_file_name and return.
    """
    model = liblinear.load_model(_cstr(model_file_name))
    if not model:
        print("can't open model file %s" % model_file_name)
        return None
    model = toPyModel(model)
    return model

def save_model(model_file_name, model, prob=None):
    """
    save_model(model_file_name, model) -> None

    Save a LIBLINEAR model to the file model_file_name.
    """
    if prob is None:
        prob = ctypes.POINTER(problem)()
    liblinear.save_model(_cstr(model_file_name), model, prob)

def train(arg1, arg2=None, arg3=None, arg4=None):
    """
    train(W, y, x [, options]) -> model | ACC

    W: a list/tuple/ndarray of l weights (type must be double).
       Use [] if no weights.

    y: a list/tuple/ndarray of l true labels (type must be int/double).

    x: 1. a list/tuple of l training instances. Feature vector of
          each training instance is a list/tuple or dictionary.

       2. an l * n numpy ndarray or scipy spmatrix (n: number of features).

    train(prob [, options]) -> model | ACC
    train(prob, param, kernel_param) -> model | ACC

    Train a model from weighted data (y, x) or a problem prob using
    'options' or a parameter param.

    If '-v' is specified in 'options' (i.e., cross validation)
    either accuracy (ACC) or mean-squared error (MSE) is returned.

    options:
        -s type : set type of solver (default 1)
          for multi-class classification
             0 -- L2-regularized logistic regression (primal)
             1 -- L2-regularized L2-loss support vector classification (dual)
             2 -- L2-regularized L2-loss support vector classification (primal)
             3 -- L2-regularized L1-loss support vector classification (dual)
             4 -- support vector classification by Crammer and Singer
             5 -- L1-regularized L2-loss support vector classification
             6 -- L1-regularized logistic regression
             7 -- L2-regularized logistic regression (dual)
          for regression
            11 -- L2-regularized L2-loss support vector regression (primal)
            12 -- L2-regularized L2-loss support vector regression (dual)
            13 -- L2-regularized L1-loss support vector regression (dual)
          for outlier detection
            21 -- one-class support vector machine (dual)
        -c cost : set the parameter C (default 1)
        -p epsilon : set the epsilon in loss function of SVR (default 0.1)
        -e epsilon : set tolerance of termination criterion
            -s 0 and 2
                |f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,
                where f is the primal function, (default 0.01)
            -s 11
                |f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.0001)
            -s 1, 3, 4, 7, and 21
                Dual maximal violation <= eps; similar to libsvm (default 0.1 except 0.01 for -s 21)
            -s 5 and 6
                |f'(w)|_inf <= eps*min(pos,neg)/l*|f'(w0)|_inf,
                where f is the primal function (default 0.01)
            -s 12 and 13
                |f'(alpha)|_1 <= eps |f'(alpha0)|,
                where f is the dual function (default 0.1)
        -B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)
        -R : not regularize the bias; must with -B 1 to have the bias; DON'T use this unless you know what it is
            (for -s 0, 2, 5, 6, 11)"
        -wi weight: weights adjust the parameter C of different classes (see README for details)
        -v n: n-fold cross validation mode
        -C : find parameters (C for -s 0, 2 and C, p for -s 11)
        -q : quiet mode (no outputs)

        [The below are used only for kernelized settings, which is supported by only a part of solvers.]
        -t kernel_type : set type of kernel function (default 0)
            0 -- linear: u'*v
            1 -- polynomial: (gamma*u'*v + coef0)^degree
            2 -- radial basis function: exp(-gamma*|u-v|^2)
            3 -- sigmoid: tanh(gamma*u'*v + coef0)
            4 -- precomputed kernel (kernel values in training_set_file)
            5 -- [for debug] linear (explicitly compute the kernel matrix)
        -d degree : set degree in kernel function (default 3)
        -g gamma : set gamma in kernel function (default 1/num_features)
        -r coef0 : set coef0 in kernel function (default 0)
        -m cachesize : set cache memory size in MB (default 100)
    """
    prob, param = None, None
    if isinstance(arg1, (list, tuple)) or (scipy and isinstance(arg1, np.ndarray)):
        assert isinstance(arg2, (list, tuple)) or (scipy and isinstance(arg2, np.ndarray))
        assert isinstance(arg3, (list, tuple)) or (scipy and isinstance(arg3, (np.ndarray, sparse.spmatrix)))
        W, y, x, options = arg1, arg2, arg3, arg4
        prob = problem(W, y, x)
        param = parameter()
        kernel_param = param.parse_options(options)
    elif isinstance(arg1, problem):
        prob = arg1
        if isinstance(arg2, parameter) and isinstance(arg3, kernel_parameter):
            param = arg2
            kernel_param = arg3
        else:
            param = parameter()
            kernel_param = param.parse_options(arg2)
    if prob == None or param == None :
        raise TypeError("Wrong types for the arguments")
    
    prob.set_bias(param.bias)
    liblinear.set_print_string_function(param.print_func)
    err_msg = liblinear.check_parameter(prob, param, kernel_param)
    if err_msg :
        raise ValueError('Error: %s' % err_msg)
    
    if kernel_param.kernel_type == LINEAR:
        kernel_param = ctypes.POINTER(kernel_parameter)()
    elif kernel_param.kernel_type == LINEAR_KERNEL:
        kernel_param.kernel_type = LINEAR

    if param.flag_find_parameters:
        nr_fold = param.nr_fold
        best_C = c_double()
        best_p = c_double()
        best_score = c_double()
        if param.flag_C_specified:
            start_C = param.C
        else:
            start_C = -1.0
        if param.flag_p_specified:
            start_p = param.p
        else:
            start_p = -1.0
        liblinear.find_parameters(prob, param, kernel_param, nr_fold, start_C, start_p, best_C, best_p, best_score)
        if param.solver_type in [L2R_LR, L2R_L2LOSS_SVC]:
            print("Best C = %g  CV accuracy = %g%%\n"% (best_C.value, 100.0*best_score.value))
        elif param.solver_type in [L2R_L2LOSS_SVR]:
            print("Best C = %g Best p = %g  CV MSE = %g\n"% (best_C.value, best_p.value, best_score.value))
        return best_C.value,best_p.value,best_score.value


    elif param.flag_cross_validation:
        l, nr_fold = prob.l, param.nr_fold
        target = (c_double * l)()
        liblinear.cross_validation(prob, param, kernel_param, nr_fold, target)
        ACC, MSE, SCC = evaluations(prob.y[:l], target[:l])
        if param.solver_type in [L2R_L2LOSS_SVR, L2R_L2LOSS_SVR_DUAL, L2R_L1LOSS_SVR_DUAL]:
            print("Cross Validation Mean squared error = %g" % MSE)
            print("Cross Validation Squared correlation coefficient = %g" % SCC)
            return MSE
        else:
            print("Cross Validation Accuracy = %g%%" % ACC)
            return ACC
    else:
        m = liblinear.train(prob, param, kernel_param)
        m = toPyModel(m)

        return m

def predict(y, x, m, options=""):
    """
    predict(y, x, m [, options]) -> (p_labels, p_acc, p_vals)

    y: a list/tuple/ndarray of l true labels (type must be int/double).
       It is used for calculating the accuracy. Use [] if true labels are
       unavailable.

    x: 1. a list/tuple of l training instances. Feature vector of
          each training instance is a list/tuple or dictionary.

       2. an l * n numpy ndarray or scipy spmatrix (n: number of features).

    Predict data (y, x) with the SVM model m.
    options:
        -b probability_estimates: whether to output probability estimates, 0 or 1 (default 0); currently for logistic regression only
        -q quiet mode (no outputs)

    The return tuple contains
    p_labels: a list of predicted labels
    p_acc: a tuple including  accuracy (for classification), mean-squared
           error, and squared correlation coefficient (for regression).
    p_vals: a list of decision values or probability estimates (if '-b 1'
            is specified). If k is the number of classes, for decision values,
            each element includes results of predicting k binary-class
            SVMs. if k = 2 and solver is not MCSVM_CS, only one decision value
            is returned. For probabilities, each element contains k values
            indicating the probability that the testing instance is in each class.
            Note that the order of classes here is the same as 'model.label'
            field in the model structure.
    """

    def info(s):
        print(s)

    if scipy and isinstance(x, np.ndarray):
        x = np.ascontiguousarray(x) # enforce row-major
    elif sparse and isinstance(x, sparse.spmatrix):
        x = x.tocsr()
    elif not isinstance(x, (list, tuple)):
        raise TypeError("type of x: {0} is not supported!".format(type(x)))

    if (not isinstance(y, (list, tuple))) and (not (scipy and isinstance(y, np.ndarray))):
        raise TypeError("type of y: {0} is not supported!".format(type(y)))

    predict_probability = 0
    argv = options.split()
    i = 0
    while i < len(argv):
        if argv[i] == '-b':
            i += 1
            predict_probability = int(argv[i])
        elif argv[i] == '-q':
            info = print_null
        else:
            raise ValueError("Wrong options")
        i+=1

    solver_type = m.param.solver_type
    nr_class = m.get_nr_class()
    nr_feature = m.get_nr_feature()
    is_prob_model = m.is_probability_model()
    bias = m.bias
    if bias >= 0:
        biasterm = feature_node(nr_feature+1, bias)
    else:
        biasterm = feature_node(-1, bias)
    pred_labels = []
    pred_values = []

    if scipy and isinstance(x, sparse.spmatrix):
        nr_instance = x.shape[0]
    else:
        nr_instance = len(x)

    if predict_probability:
        if not is_prob_model:
            raise TypeError('probability output is only supported for logistic regression')
        prob_estimates = (c_double * nr_class)()
        for i in range(nr_instance):
            if scipy and isinstance(x, sparse.spmatrix):
                indslice = slice(x.indptr[i], x.indptr[i+1])
                xi, idx = gen_feature_nodearray((x.indices[indslice], x.data[indslice]), feature_max=nr_feature)
            else:
                xi, idx = gen_feature_nodearray(x[i], feature_max=nr_feature)
            xi[-2] = biasterm
            label = liblinear.predict_probability(m, xi, prob_estimates)
            values = prob_estimates[:nr_class]
            pred_labels += [label]
            pred_values += [values]
    else:
        if nr_class <= 2:
            nr_classifier = 1
        else:
            nr_classifier = nr_class
        dec_values = (c_double * nr_classifier)()
        for i in range(nr_instance):
            if scipy and isinstance(x, sparse.spmatrix):
                indslice = slice(x.indptr[i], x.indptr[i+1])
                xi, idx = gen_feature_nodearray((x.indices[indslice], x.data[indslice]), feature_max=nr_feature)
            else:
                xi, idx = gen_feature_nodearray(x[i], feature_max=nr_feature)
            xi[-2] = biasterm
            label = liblinear.predict_values(m, xi, dec_values)
            values = dec_values[:nr_classifier]
            pred_labels += [label]
            pred_values += [values]

    if len(y) == 0:
        y = [0] * nr_instance
    ACC, MSE, SCC = evaluations(y, pred_labels)

    if m.is_regression_model():
        info("Mean squared error = %g (regression)" % MSE)
        info("Squared correlation coefficient = %g (regression)" % SCC)
    else:
        info("Accuracy = %g%% (%d/%d) (classification)" % (ACC, int(round(nr_instance*ACC/100)), nr_instance))

    return pred_labels, (ACC, MSE, SCC), pred_values
