#!/usr/bin/env python

from ctypes import *
from ctypes.util import find_library
from os import path
from glob import glob
import sys

try:
    import numpy as np
    import scipy
    from scipy import sparse
except:
    scipy = None

if sys.version_info[0] < 3:
    range = xrange
    from itertools import izip as zip

__all__ = ['liblinear', 'feature_node', 'gen_feature_nodearray', 'problem',
           'parameter', 'model', 'toPyModel', 'L2R_LR', 'L2R_L2LOSS_SVC_DUAL',
           'L2R_L2LOSS_SVC', 'L2R_L1LOSS_SVC_DUAL', 'MCSVM_CS',
           'L1R_L2LOSS_SVC', 'L1R_LR', 'L2R_LR_DUAL', 'L2R_L2LOSS_SVR',
           'L2R_L2LOSS_SVR_DUAL', 'L2R_L1LOSS_SVR_DUAL', 'ONECLASS_SVM',
           'print_null'] + ['kernel_parameter', 'LINEAR', 'POLY', 'RBF', 'SIGMOID', 'PRECOMPUTED', 'LINEAR_KERNEL'] # The latter list is needed for kernels

try:
    dirname = path.dirname(path.abspath(__file__))
    dynamic_lib_name = 'clib.cp*'
    path_to_so = glob(path.join(dirname, dynamic_lib_name))[0]
    liblinear = CDLL(path_to_so)
except:
    try :
        if sys.platform == 'win32':
            liblinear = CDLL(path.join(dirname, r'..\..\windows\liblinear.dll'))
        else:
            liblinear = CDLL(path.join(dirname, '../../liblinear.so.5'))
    except:
    # For unix the prefix 'lib' is not considered.
        if find_library('linear'):
            liblinear = CDLL(find_library('linear'))
        elif find_library('liblinear'):
            liblinear = CDLL(find_library('liblinear'))
        else:
            raise Exception('LIBLINEAR library not found.')

L2R_LR = 0
L2R_L2LOSS_SVC_DUAL = 1
L2R_L2LOSS_SVC = 2
L2R_L1LOSS_SVC_DUAL = 3
MCSVM_CS = 4
L1R_L2LOSS_SVC = 5
L1R_LR = 6
L2R_LR_DUAL = 7
L2R_L2LOSS_SVR = 11
L2R_L2LOSS_SVR_DUAL = 12
L2R_L1LOSS_SVR_DUAL = 13
ONECLASS_SVM = 21

LINEAR = 0
POLY = 1
RBF = 2
SIGMOID = 3
PRECOMPUTED = 4
LINEAR_KERNEL = 5

PRINT_STRING_FUN = CFUNCTYPE(None, c_char_p)
def print_null(s):
    return

def genFields(names, types):
    return list(zip(names, types))

def fillprototype(f, restype, argtypes):
    f.restype = restype
    f.argtypes = argtypes

class feature_node(Structure):
    _names = ["index", "value"]
    _types = [c_int, c_double]
    _fields_ = genFields(_names, _types)

    def __str__(self):
        return '%d:%g' % (self.index, self.value)

def gen_feature_nodearray(xi, feature_max=None):
    if feature_max:
        assert(isinstance(feature_max, int))

    xi_shift = 0 # ensure correct indices of xi
    if scipy and isinstance(xi, tuple) and len(xi) == 2\
            and isinstance(xi[0], np.ndarray) and isinstance(xi[1], np.ndarray): # for a sparse vector
        index_range = xi[0] + 1 # index starts from 1
        if feature_max:
            index_range = index_range[np.where(index_range <= feature_max)]
    elif scipy and isinstance(xi, np.ndarray):
        xi_shift = 1
        index_range = xi.nonzero()[0] + 1 # index starts from 1
        if feature_max:
            index_range = index_range[np.where(index_range <= feature_max)]
    elif isinstance(xi, (dict, list, tuple)):
        if isinstance(xi, dict):
            index_range = xi.keys()
        elif isinstance(xi, (list, tuple)):
            xi_shift = 1
            index_range = range(1, len(xi) + 1)
        index_range = filter(lambda j: xi[j-xi_shift] != 0, index_range)

        if feature_max:
            index_range = filter(lambda j: j <= feature_max, index_range)
        index_range = sorted(index_range)
    else:
        raise TypeError('xi should be a dictionary, list, tuple, 1-d numpy array, or tuple of (index, data)')

    ret = (feature_node*(len(index_range)+2))()
    ret[-1].index = -1 # for bias term
    ret[-2].index = -1

    if scipy and isinstance(xi, tuple) and len(xi) == 2\
            and isinstance(xi[0], np.ndarray) and isinstance(xi[1], np.ndarray): # for a sparse vector
        for idx, j in enumerate(index_range):
            ret[idx].index = j
            ret[idx].value = (xi[1])[idx]
    else:
        for idx, j in enumerate(index_range):
            ret[idx].index = j
            ret[idx].value = xi[j - xi_shift]

    max_idx = 0
    if len(index_range) > 0:
        max_idx = index_range[-1]
    return ret, max_idx

try:
    from numba import jit
    jit_enabled = True
except:
    jit = lambda x: x
    jit_enabled = False

@jit
def csr_to_problem_jit(l, x_val, x_ind, x_rowptr, prob_val, prob_ind, prob_rowptr):
    for i in range(l):
        b1,e1 = x_rowptr[i], x_rowptr[i+1]
        b2,e2 = prob_rowptr[i], prob_rowptr[i+1]-2
        for j in range(b1,e1):
            prob_ind[j-b1+b2] = x_ind[j]+1
            prob_val[j-b1+b2] = x_val[j]
def csr_to_problem_nojit(l, x_val, x_ind, x_rowptr, prob_val, prob_ind, prob_rowptr):
    for i in range(l):
        x_slice = slice(x_rowptr[i], x_rowptr[i+1])
        prob_slice = slice(prob_rowptr[i], prob_rowptr[i+1]-2)
        prob_ind[prob_slice] = x_ind[x_slice]+1
        prob_val[prob_slice] = x_val[x_slice]

def csr_to_problem(x, prob):
    # Extra space for termination node and (possibly) bias term
    x_space = prob.x_space = np.empty((x.nnz+x.shape[0]*2), dtype=feature_node)
    # rowptr has to be a 64bit integer because it will later be used for pointer arithmetic,
    # which overflows when the added pointer points to an address that is numerically high.
    prob.rowptr = x.indptr.astype(np.int64, copy=True)
    prob.rowptr[1:] += 2*np.arange(1,x.shape[0]+1)
    prob_ind = x_space["index"]
    prob_val = x_space["value"]
    prob_ind[:] = -1
    if jit_enabled:
        csr_to_problem_jit(x.shape[0], x.data, x.indices, x.indptr, prob_val, prob_ind, prob.rowptr)
    else:
        csr_to_problem_nojit(x.shape[0], x.data, x.indices, x.indptr, prob_val, prob_ind, prob.rowptr)

class problem(Structure):
    _names = ["l", "n", "y", "x", "bias", "W"]
    _types = [c_int, c_int, POINTER(c_double), POINTER(POINTER(feature_node)), c_double, POINTER(c_double)]
    _fields_ = genFields(_names, _types)

    def __init__(self, W, y, x, bias = -1):
        if (not isinstance(W, (list, tuple))) and (not (scipy and isinstance(W, np.ndarray))):
            raise TypeError("type of W: {0} is not supported!".format(type(W)))
        if (not isinstance(y, (list, tuple))) and (not (scipy and isinstance(y, np.ndarray))):
            raise TypeError("type of y: {0} is not supported!".format(type(y)))

        if isinstance(x, (list, tuple)):
            if len(y) != len(x):
                raise ValueError("len(y) != len(x)")
            if len(W) != 0 and len(W) != len(x):
                raise ValueError("len(W) != len(x)")
        elif scipy != None and isinstance(x, (np.ndarray, sparse.spmatrix)):
            if len(y) != x.shape[0]:
                raise ValueError("len(y) != len(x)")
            if len(W) != 0 and len(W) != x.shape[0]:
                raise ValueError("len(W) != len(x)")
            if isinstance(x, np.ndarray):
                x = np.ascontiguousarray(x) # enforce row-major
            if isinstance(x, sparse.spmatrix):
                x = x.tocsr()
                pass
        else:
            raise TypeError("type of x: {0} is not supported!".format(type(x)))
        self.l = l = len(y)
        self.bias = -1
        if len(W) == 0:
            W = [1] * l

        max_idx = 0
        x_space = self.x_space = []
        if scipy != None and isinstance(x, sparse.csr_matrix):
            csr_to_problem(x, self)
            max_idx = x.shape[1]
        else:
            for i, xi in enumerate(x):
                tmp_xi, tmp_idx = gen_feature_nodearray(xi)
                x_space += [tmp_xi]
                max_idx = max(max_idx, tmp_idx)
        self.n = max_idx

        self.W = (c_double * l)()
        #if scipy != None and isinstance(W, scipy.ndarray):
        if scipy != None and isinstance(W, np.ndarray):
            #scipy.ctypeslib.as_array(self.W, (self.l,))[:] = W
            np.ctypeslib.as_array(self.W, (self.l,))[:] = W
        else:
            for i, Wi in enumerate(W): self.W[i] = Wi

        self.y = (c_double * l)()
        if scipy != None and isinstance(y, np.ndarray):
            np.ctypeslib.as_array(self.y, (self.l,))[:] = y
        else:
            for i, yi in enumerate(y): self.y[i] = yi

        self.x = (POINTER(feature_node) * l)()
        if scipy != None and isinstance(x, sparse.csr_matrix):
            base = addressof(self.x_space.ctypes.data_as(POINTER(feature_node))[0])
            x_ptr = cast(self.x, POINTER(c_uint64))
            x_ptr = np.ctypeslib.as_array(x_ptr,(self.l,))
            x_ptr[:] = self.rowptr[:-1]*sizeof(feature_node)+base
        else:
            for i, xi in enumerate(self.x_space): self.x[i] = xi

        self.set_bias(bias)

    def set_bias(self, bias):
        if self.bias == bias:
            return
        if bias >= 0 and self.bias < 0:
            self.n += 1
            node = feature_node(self.n, bias)
        if bias < 0 and self.bias >= 0:
            self.n -= 1
            node = feature_node(-1, bias)

        if isinstance(self.x_space, list):
            for xi in self.x_space:
                xi[-2] = node
        else:
            self.x_space["index"][self.rowptr[1:]-2] = node.index
            self.x_space["value"][self.rowptr[1:]-2] = node.value

        self.bias = bias

class kernel_parameter(Structure):
    _names = ["kernel_type", "degree", "gamma", "coef0", "cache_size"]
    _types = [c_int, c_int, c_double, c_double, c_double]
    _fields_ = genFields(_names, _types)
    
    def __init__(self):
        pass
    
    def __str__(self):
        s = ''
        attrs = parameter._names + list(self.__dict__.keys())
        values = map(lambda attr: getattr(self, attr), attrs)
        for attr, val in zip(attrs, values):
            s += (' %s: %s\n' % (attr, val))
        s = s.strip()

        return s

    def set_to_default_values(self):
        self.kernel_type = 0
        self.degree = 3
        self.gamma = 0
        self.coef0 = 0
        self.cache_size = 100
    
    



class parameter(Structure):
    _names = ["solver_type", "eps", "C", "nr_weight", "weight_label", "weight", "p", "nu", "init_sol", "regularize_bias"]
    _types = [c_int, c_double, c_double, c_int, POINTER(c_int), POINTER(c_double), c_double, c_double, POINTER(c_double), c_int]
    _fields_ = genFields(_names, _types)

    def __init__(self, options = None):
        if options == None:
            options = ''
        self.parse_options(options)

    def __str__(self):
        s = ''
        attrs = parameter._names + list(self.__dict__.keys())
        values = map(lambda attr: getattr(self, attr), attrs)
        for attr, val in zip(attrs, values):
            s += (' %s: %s\n' % (attr, val))
        s = s.strip()

        return s

    def set_to_default_values(self):
        self.solver_type = L2R_L2LOSS_SVC_DUAL
        self.eps = float('inf')
        self.C = 1
        self.p = 0.1
        self.nu = 0.5
        self.nr_weight = 0
        self.weight_label = None
        self.weight = None
        self.init_sol = None
        self.bias = -1
        self.regularize_bias = 1
        self.flag_cross_validation = False
        self.flag_C_specified = False
        self.flag_p_specified = False
        self.flag_solver_specified = False
        self.flag_find_parameters = False
        self.nr_fold = 0
        self.print_func = cast(None, PRINT_STRING_FUN)

    def parse_options(self, options):
        if isinstance(options, list):
            argv = options
        elif isinstance(options, str):
            argv = options.split()
        else:
            raise TypeError("arg 1 should be a list or a str.")
        self.set_to_default_values()
        self.print_func = cast(None, PRINT_STRING_FUN)
        weight_label = []
        weight = []
        
        kernel_param = kernel_parameter()
        kernel_param.set_to_default_values()

        i = 0
        while i < len(argv) :
            if argv[i] == "-s":
                i = i + 1
                self.solver_type = int(argv[i])
                self.flag_solver_specified = True
            elif argv[i] == "-c":
                i = i + 1
                self.C = float(argv[i])
                self.flag_C_specified = True
            elif argv[i] == "-p":
                i = i + 1
                self.p = float(argv[i])
                self.flag_p_specified = True
            elif argv[i] == "-n":
                i = i + 1
                self.nu = float(argv[i])
            elif argv[i] == "-e":
                i = i + 1
                self.eps = float(argv[i])
            elif argv[i] == "-B":
                i = i + 1
                self.bias = float(argv[i])
            elif argv[i] == "-v":
                i = i + 1
                self.flag_cross_validation = 1
                self.nr_fold = int(argv[i])
                if self.nr_fold < 2 :
                    raise ValueError("n-fold cross validation: n must >= 2")
            elif argv[i].startswith("-w"):
                i = i + 1
                self.nr_weight += 1
                weight_label += [int(argv[i-1][2:])]
                weight += [float(argv[i])]
            elif argv[i] == "-q":
                self.print_func = PRINT_STRING_FUN(print_null)
            elif argv[i] == "-C":
                self.flag_find_parameters = True
            elif argv[i] == "-R":
                self.regularize_bias = 0
            elif argv[i] == "-t":
                i = i + 1
                kernel_param.kernel_type = int(argv[i])
            elif argv[i] == "-d":
                i = i + 1
                kernel_param.degree = int(argv[i])
            elif argv[i] == "-g":
                i = i + 1
                kernel_param.gamma = float(argv[i])
            elif argv[i] == "-r":
                i = i + 1
                kernel_param.coef0 = float(argv[i])
            elif argv[i] == "-m":
                i = i + 1
                kernel_param.cache_size = float(argv[i])
            else:
                raise ValueError("Wrong options")
            i += 1

        liblinear.set_print_string_function(self.print_func)
        self.weight_label = (c_int*self.nr_weight)()
        self.weight = (c_double*self.nr_weight)()
        for i in range(self.nr_weight):
            self.weight[i] = weight[i]
            self.weight_label[i] = weight_label[i]

        # default solver for parameter selection is L2R_L2LOSS_SVC
        if self.flag_find_parameters:
            if not self.flag_cross_validation:
                self.nr_fold = 5
            if not self.flag_solver_specified:
                self.solver_type = L2R_L2LOSS_SVC
                self.flag_solver_specified = True
            elif self.solver_type not in [L2R_LR, L2R_L2LOSS_SVC, L2R_L2LOSS_SVR]:
                raise ValueError("Warm-start parameter search only available for -s 0, -s 2 and -s 11")

        if self.eps == float('inf'):
            if self.solver_type in [L2R_LR, L2R_L2LOSS_SVC]:
                self.eps = 0.01
            elif self.solver_type in [L2R_L2LOSS_SVR]:
                self.eps = 0.0001
            elif self.solver_type in [L2R_L2LOSS_SVC_DUAL, L2R_L1LOSS_SVC_DUAL, MCSVM_CS, L2R_LR_DUAL]:
                self.eps = 0.1
            elif self.solver_type in [L1R_L2LOSS_SVC, L1R_LR]:
                self.eps = 0.01
            elif self.solver_type in [L2R_L2LOSS_SVR_DUAL, L2R_L1LOSS_SVR_DUAL]:
                self.eps = 0.1
            elif self.solver_type in [ONECLASS_SVM]:
                self.eps = 0.01
        
        return kernel_param

class model(Structure):
    _names = ["param", "kernelized", "kernel_param", "nr_class", "nr_feature", "nr_feature_orig", "w", "label", "bias", "rho", "SV", "y"]
    _types = [parameter, c_int, kernel_parameter, c_int, c_int, c_int, POINTER(c_double), POINTER(c_int), c_double, c_double, POINTER(POINTER(feature_node)), POINTER(c_double)]
    _fields_ = genFields(_names, _types)

    def __init__(self):
        self.__createfrom__ = 'python'

    def __del__(self):
        # free memory created by C to avoid memory leak
        if hasattr(self, '__createfrom__') and self.__createfrom__ == 'C':
            liblinear.free_and_destroy_model(pointer(self))

    def get_nr_feature(self):
        return liblinear.get_nr_feature(self)

    def get_nr_class(self):
        return liblinear.get_nr_class(self)

    def get_labels(self):
        nr_class = self.get_nr_class()
        labels = (c_int * nr_class)()
        liblinear.get_labels(self, labels)
        return labels[:nr_class]

    def get_decfun_coef(self, feat_idx, label_idx=0):
        return liblinear.get_decfun_coef(self, feat_idx, label_idx)

    def get_decfun_bias(self, label_idx=0):
        return liblinear.get_decfun_bias(self, label_idx)

    def get_decfun_rho(self):
        return liblinear.get_decfun_rho(self)

    def get_decfun(self, label_idx=0):
        w = [liblinear.get_decfun_coef(self, feat_idx, label_idx) for feat_idx in range(1, self.nr_feature+1)]
        if self.is_oneclass_model():
            rho = self.get_decfun_rho()
            return (w, -rho)
        else:
            b = liblinear.get_decfun_bias(self, label_idx)
            return (w, b)

    def is_probability_model(self):
        return (liblinear.check_probability_model(self) == 1)

    def is_regression_model(self):
        return (liblinear.check_regression_model(self) == 1)

    def is_oneclass_model(self):
        return (liblinear.check_oneclass_model(self) == 1)

def toPyModel(model_ptr):
    """
    toPyModel(model_ptr) -> model

    Convert a ctypes POINTER(model) to a Python model
    """
    if bool(model_ptr) == False:
        raise ValueError("Null pointer")
    m = model_ptr.contents
    m.__createfrom__ = 'C'
    return m

fillprototype(liblinear.train, POINTER(model), [POINTER(problem), POINTER(parameter), POINTER(kernel_parameter)])
fillprototype(liblinear.find_parameters, None, [POINTER(problem), POINTER(parameter), POINTER(kernel_parameter), c_int, c_double, c_double, POINTER(c_double), POINTER(c_double), POINTER(c_double)])
fillprototype(liblinear.cross_validation, None, [POINTER(problem), POINTER(parameter), POINTER(kernel_parameter), c_int, POINTER(c_double)])

fillprototype(liblinear.predict_values, c_double, [POINTER(model), POINTER(feature_node), POINTER(c_double)])
fillprototype(liblinear.predict, c_double, [POINTER(model), POINTER(feature_node)])
fillprototype(liblinear.predict_probability, c_double, [POINTER(model), POINTER(feature_node), POINTER(c_double)])

fillprototype(liblinear.save_model, c_int, [c_char_p, POINTER(model), POINTER(problem)])
fillprototype(liblinear.load_model, POINTER(model), [c_char_p])

fillprototype(liblinear.get_nr_feature, c_int, [POINTER(model)])
fillprototype(liblinear.get_nr_class, c_int, [POINTER(model)])
fillprototype(liblinear.get_labels, None, [POINTER(model), POINTER(c_int)])
fillprototype(liblinear.get_decfun_coef, c_double, [POINTER(model), c_int, c_int])
fillprototype(liblinear.get_decfun_bias, c_double, [POINTER(model), c_int])
fillprototype(liblinear.get_decfun_rho, c_double, [POINTER(model)])

fillprototype(liblinear.free_model_content, None, [POINTER(model)])
fillprototype(liblinear.free_and_destroy_model, None, [POINTER(POINTER(model))])
fillprototype(liblinear.destroy_param, None, [POINTER(parameter)])
fillprototype(liblinear.check_parameter, c_char_p, [POINTER(problem), POINTER(parameter), POINTER(kernel_parameter)])
fillprototype(liblinear.check_probability_model, c_int, [POINTER(model)])
fillprototype(liblinear.check_regression_model, c_int, [POINTER(model)])
fillprototype(liblinear.check_oneclass_model, c_int, [POINTER(model)])
fillprototype(liblinear.set_print_string_function, None, [CFUNCTYPE(None, c_char_p)])
