import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import time
import sys
import csv

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision
#from jax import random
import functools
#from jax import jit, grad, vmap
#import neural_tangents as nt
#from neural_tangents import stax

sys.path.append('./source/python/build/lib.linux-x86_64-cpython-312')
from liblinear import liblinear, liblinearutil, commonutil
import find_invsq
import maximize_quadratic as mq



def numform(value, digits):
    s = str(value)
    pos = s.find('.')
    if pos == -1:
        if len(s) <= digits - 2:
            return s + '.0'
        else:
            return s
    else:
        if pos >= digits - 1:
            return s[:pos]
        else:
            return s[:(digits+1)]


def prepare_train_svm(x, y, weight):
    if len(y.shape) != 1:
        raise RuntimeError(f'`y` must be a vector (given shape {y.shape})')
    
    if x is not None:
        if len(x.shape) != 2:
            raise RuntimeError(f'`x` must be a matrix (given shape {x.shape})')
    
        n, d = x.shape
        if y.size != n:
            raise RuntimeError(f'`y` must be a vector of size {n} (given shape {y.shape})')
    else:
        d = None
        n = y.size
    
    if weight.shape != (n,):
        raise RuntimeError(f'Size of `weight` must be the number of training samples (given shape {weight.shape})')

    return n, d, x, y, weight


def linear_kernel(x1, x2):
    k = np.dot(x1, x2.T)
    return k


def rbf_kernel(x1, x2):
    dist = np.array([np.linalg.norm(x1[i,:] - x2, axis=1) for i in range(x1.shape[0])])
    gamma = 1 / (x_train.shape[1] * x_train.var())
    k = np.exp(-gamma * dist**2)
    return k


def cal_ntk(X_train, X_test,
    depth=2,
    width=1024,
    W_std = np.sqrt(2), 
    b_std = 0.1,
    num_classes = 1,
    parameterization = 'ntk',
    activation = 'relu'):
    
    activation_fn = stax.Relu()
    dense = functools.partial(
        stax.Dense, W_std=W_std, b_std=b_std, parameterization=parameterization)

    layers = [stax.Flatten()]
    for _ in range(depth):
        layers += [dense(width), activation_fn]
    layers += [stax.Dense(num_classes, W_std=W_std, b_std=b_std, 
                            parameterization=parameterization)]

    init_fn, apply_fn, kernel_fn = stax.serial(*layers)
    kernel_fn = jit(kernel_fn, static_argnums=(2,))
    ntk_train_test = kernel_fn(X_train, X_test, 'ntk')
    ntk_train_test = np.array(ntk_train_test)
    
    print('NTK calculation done!')
    return ntk_train_test  


def cal_cntk(X_train, X_test,
    depth=2,
    width=1024,
    W_std = np.sqrt(2), 
    b_std = 0.1,
    num_classes = 1,
    parameterization = 'ntk',
    activation = 'relu'):
  
    activation_fn = stax.Relu()
    conv = functools.partial(
        stax.Conv,
        W_std=W_std,
        b_std=b_std,
        padding='SAME',
        parameterization=parameterization)
    layers = [conv(width, (3,3)), activation_fn]
    for _ in range(depth-1):
        layers += [conv(width, (3,3)), activation_fn]
    layers += [stax.Flatten(), stax.Dense(num_classes, W_std=W_std, b_std=b_std,
                                            parameterization=parameterization)]
    init_fn, apply_fn, kernel_fn = stax.serial(*layers)
    kernel_fn = jit(kernel_fn, static_argnums=(2,))
    X_train = np.transpose(X_train, (0, 2, 3, 1))
    X_test = np.transpose(X_test, (0, 2, 3, 1))
    ntk_train_test = kernel_fn(X_train, X_test, 'ntk')
    ntk_train_test = np.array(ntk_train_test)  
    print('NTK calculation done!')
    return ntk_train_test  


def train_kernel_dual(kernel_matrix, y, lam, weight):
    n, _, _, y, weight = prepare_train_svm(None, y, weight)
    if kernel_matrix.shape != (n, n):
        raise RuntimeError(f'`kernel_matrix` must be a matrix of size {n}x{n}')
    kernel_matrix_ll = scipy.sparse.csr_matrix(np.hstack((np.arange(1, n+1).reshape((-1, 1)), kernel_matrix)))

    # Since LIBLINEAR sets the strength of regularization as (1/2)||w||_2^2 + C Σ w_i loss_i,
    # so C = 1 / lam
    c = 1.0 / lam
    
    # model = liblinearutil.train(weight, y, kernel_matrix_ll, f'-s 3 -t 4 -c {c} -e 0.000001') # SVM
    model = liblinearutil.train(weight, y, kernel_matrix_ll, f'-s 7 -t 4 -c {c} -e 0.000001') # Logistic
    alpha_ll, _ = model.get_decfun()
    alpha_ll = np.array(alpha_ll)
    return alpha_ll / (c * weight)


def predict(x_train, x_test, y_train, lam, weight, alpha):
    n = x_train.shape[0]
    k = kernel(x_train, x_test)
    wyak = (weight * y_train * alpha).reshape(-1,1) * k
    pred = (1 / lam) * np.sum(wyak, axis=0)
    ypred = np.where(pred > 0, 1, -1)
    return pred, ypred


def score(ypred, y_test):
    n_correct = np.sum(ypred==y_test)
    test_score = n_correct / len(ypred)
    return test_score, n_correct


def extract_by_label(x, y, l_pos, l_neg):
    used_instances = (y == l_pos) | (y == l_neg)
    xsub = x[used_instances]
    ysub = np.where(y[used_instances] == l_pos, 1, -1)
    return xsub, ysub

# maximize DG with respect to training weight
def quad_maxim(pr, hg, v_new, wya, k, weight_dist, orig_weight, lam):
    n = k.shape[0]
    h = hg * v_new
    s = (1/lam) * ((wya * v_new) * k.T).T * (wya * v_new)
    s = (s + s.T) / 2
    
    # sum constraint
    eta = np.ones(n)
    tau = np.dot(eta, orig_weight)
    
    # maximize DG
    mqresult = mq.maximize_l2_linear_rawmatrix(s, h, orig_weight, weight_dist, eta, tau, tolerance=1.0e-12, details=True)
    res = max(mqresult, key=lambda r: r['value'])
    max_gap_new = (res['value'] / 2) + pr
    worst_weight = res['weights']
    return max_gap_new, worst_weight

# minimize accuracy with respect to validation weight
def opt_acc(n_correct, y_test, weight_change_base):
    # define weight dist
    weight = np.array([1.0] * len(y_test))
    new_weight = weight * (weight_change_base * (y_test + 1.0) * 0.5 - (y_test - 1.0) * 0.5)
    weight_dist = np.linalg.norm(new_weight - weight)

    binary = np.array([1.] * n_correct + [0.] * (len(y_test) - n_correct))
    m = len(binary)
    w_tilde = np.array([1.0] * m)

    binary_sum = np.sum(binary)
    sqrt_term = np.sqrt(np.linalg.norm(binary)**2 - binary_sum**2 / m)
    sqrt_term = max(sqrt_term, 1e-8)

    # weighted accuracy
    acc_w = (np.dot(binary, w_tilde) - weight_dist * sqrt_term) / np.sum(w_tilde)
    w_te = w_tilde - weight_dist * (binary_sum / m * np.ones_like(binary) - binary) / sqrt_term
    return acc_w, w_te


def remove_greedy1(x, y, x_test, y_test, lam, alpha, weight, weight_dist, weight_change_base, pred):
    n = x.shape[0]
    k = kernel(x, x)
    
    # constant c (:pr)
    wya = weight * y * alpha
    pr = (0.5 / lam) * np.dot(np.dot(wya, k), wya)
    # vetor b (:hg*v)
    # hg = np.maximum(0.0, 1.0 - y * pred) - alpha    # SVM
    hg = np.log(1 + np.exp(- y * pred)) + (1 - alpha) * np.log(1 - alpha) + alpha * np.log(alpha)     # Logistic
    # set decision vector v
    v_initial = np.ones(n)
    v_current = v_initial.copy()
    # matrix A (:pr_conj)
    Kwya = np.outer(wya, wya) * k
    pr_conj = (0.5 / lam) * v_initial.T @ Kwya @ v_initial
    # initial DG
    h = hg @ v_initial
    ini_value = pr + h + pr_conj

    r_list = []
    acc_list = []
    worst_weight_list = []
    
    for iter in range(n):  # repeat for the number of removals
        weight_list = []    # worst-case acc while relearning
        values = []
        v_possible = []
            
        ################ selection ################
        t_st = time.time()
        if weight_dist == 0.0:
            worst_weight = np.ones(len(v_current))
            
            # before selection
            if np.sum(v_current) == n and iter == 0:
                values.append(ini_value)
                weight_list.append(worst_weight)
                v_possible.append(v_initial) 
            else:
                for i in range(len(v_current)):
                    if v_current[i] == 1:
                        v_new = v_current.copy()
                        v_new[i] = 0
                        v_possible.append(v_new)

                        # DG
                        pr_conj = (0.5 / lam) * v_new.T @ Kwya @ v_new
                        h = hg @ v_new
                        value = pr + h + pr_conj
                        values.append(value)
                        weight_list.append(worst_weight)
        else:
            # before selection
            if np.sum(v_current) == n and iter == 0:
                max_gap, worst_weight = quad_maxim(pr, hg, v_initial, wya, k, weight_dist, weight, lam)
                values.append(max_gap)
                weight_list.append(worst_weight)
                v_possible.append(v_initial)
            else:
                for i in range(len(v_current)):
                    if v_current[i] == 1:
                        v_new = v_current.copy()
                        v_new[i] = 0
                        v_possible.append(v_new)    
        
                        # DG
                        max_gap_new, worst_weight = quad_maxim(pr, hg, v_new, wya, k, weight_dist, weight, lam)
                        values.append(max_gap_new)
                        weight_list.append(worst_weight)
                        
        # find decision vector v
        best_index = np.argmin(values)
        min_value = values[best_index]
        worst_weight = weight_list[best_index]
        v_best = v_possible[best_index]
        v_current = v_best  # retain v for next loop
        
        print(f'{min_value=}')
        t_en = time.time()
        t_run = t_en - t_st
        print(f'{t_run=}')

        radii_here = (2.0*min_value/lam) ** 0.5
        ex_ind = [i for i, ele in enumerate(v_best) if ele == 1]
        rm_ind = list(set(range(n))-set(ex_ind))
        
        summary = f"\n========instance removals========\n" \
                f"\n n_rm={len(rm_ind)}" \
                f"\n========radius when some instances are removed========\n" \
                f"{radii_here=}\n"
    
        print(summary)
        worst_weight_list.append(worst_weight)
        r_list.append(radii_here)

        ################ relearning ################
        acc_w = relearning(x, y, x_test, y_test, v_current, worst_weight, lam, n, weight_change_base)
        acc_list.append(acc_w)
    return r_list, acc_list, worst_weight_list


def remove_greedy2(x, y, x_test, y_test, lam, alpha, weight, weight_dist, weight_change_base, pred):
    n = x.shape[0]
    k = kernel(x, x)

    # for pr
    wya = weight * y * alpha
    pr = (0.5 / lam) * np.dot(np.dot(wya, k), wya)
    # for h
    # hg = np.maximum(0.0, 1.0 - y * pred) - alpha    # SVM
    hg = np.log(1 + np.exp(- y * pred)) + (1 - alpha) * np.log(1 - alpha) + alpha * np.log(alpha)     # Logistic
    
    # for pr_conj
    Kwya = np.outer(wya, wya) * k

    # decision variable
    v_initial = np.ones(n)
    v_current = v_initial.copy()

    # calcuration of worst-case weight
    if weight_dist == 0.0:
        worst_weight = np.ones(n)
    else:
        _, worst_weight = quad_maxim(pr, hg, v_initial, wya, k, weight_dist, weight, lam)

    r_list = []
    acc_list = []

    for i in range(n):
        ################ relearning ################
        acc_w = relearning(x, y, x_test, y_test, v_current, worst_weight, lam, n, weight_change_base)
        acc_list.append(acc_w)

        values = []
        v_possible = []

        ################ selection ################
        t_st = time.time()
        for i in range(n):
            if v_current[i] == 1:
                v_new = v_current.copy()
                v_new[i] = 0
                v_possible.append(v_new)

                # DualityGap
                pr_conj = (0.5 / lam) * (worst_weight * v_new).T @ Kwya @ (worst_weight * v_new)
                h = hg @ (worst_weight * v_new)
                value = pr + h + pr_conj
                values.append(value)

        # find decision vector v
        best_index = np.argmin(values)
        min_value = values[best_index]
        v_best = v_possible[best_index]
        v_current = v_best  # retain v for next loop
        
        t_en = time.time()
        t_run = t_en - t_st
        print(f'TIME(EACH SELECTION): {t_run=}')

        radii_here = (2.0*min_value/lam) ** 0.5
        ex_ind = [i for i, ele in enumerate(v_best) if ele == 1]
        rm_ind = list(set(range(n))-set(ex_ind))
        
        summary = f"\n n_rm={len(rm_ind)}" \
                f"\n========radius when some instances are removed========\n" \
                f"{radii_here=}\n"
        print(summary)
        r_list.append(radii_here)
    
    return r_list, acc_list, worst_weight


def remove_greedy3(x, y, x_test, y_test, lam, alpha, weight, weight_dist, weight_change_base, pred):
    n = x.shape[0]
    k = kernel(x, x)

    # for pr
    wya = weight * y * alpha
    pr = (0.5 / lam) * np.dot(np.dot(wya, k), wya)
    # for h
    # hg = np.maximum(0.0, 1.0 - y * pred) - alpha    # SVM
    hg = np.log(1 + np.exp(- y * pred)) + (1 - alpha) * np.log(1 - alpha) + alpha * np.log(alpha)     # Logistic
    # for pr_conj
    Kwya = np.outer(wya, wya) * k

    # decision variable
    v_initial = np.ones(n)
    v_current = v_initial.copy()

    # calcuration of worst-case weight
    if weight_dist == 0.0:
        worst_weight = np.ones(n)
    else:
        _, worst_weight = quad_maxim(pr, hg, v_initial, wya, k, weight_dist, weight, lam)

    values = []
    v_possible = []

    ################ selection ################
    t_st = time.time()
    for i in range(n):
        if v_current[i] == 1:
            v_new = v_current.copy()
            v_new[i] = 0
            v_possible.append(v_new)

            # DualityGap
            pr_conj = (0.5 / lam) * (worst_weight * v_new).T @ Kwya @ (worst_weight * v_new)
            h = hg @ (worst_weight * v_new)
            value = pr + h + pr_conj
            values.append(value)
            # print(f'{i=}, {value=}')
    
    sort_index = np.argsort(values)
    t_en = time.time()
    t_run = t_en - t_st
    print(f'TIME(SELECTION): {t_run=}')
    
    # calculate r_list and acc
    r_list = []
    acc_list = []
    v_new = np.ones(n)

    fraclist = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 1.0]
    checkpoint_list = [int(len(x) * frac) for frac in fraclist]

    for num_select in range(n-1):
        ################ relearning ################
        n_current = np.sum(v_new)
        if n_current in checkpoint_list:
            acc_w = relearning(x, y, x_test, y_test, v_new, worst_weight, lam, n, weight_change_base)
            acc_list.append(acc_w)
        
        # remove
        remove_index = sort_index[:(num_select+1)]
        v_new[remove_index] = 0

        # DualityGap
        pr_conj = (0.5 / lam) * (worst_weight * v_new).T @ Kwya @ (worst_weight * v_new)
        h = hg @ (worst_weight * v_new)
        min_value = pr + h + pr_conj
        values.append(value)

        radii_here = (2.0*min_value/lam) ** 0.5
        ex_ind = [i for i, ele in enumerate(v_new) if ele == 1]
        rm_ind = list(set(range(n))-set(ex_ind))
        
        summary = f"\n n_rm={len(rm_ind)}" \
                f"\n========radius when some instances are removed========\n" \
                f"{radii_here=}\n"
        print(summary)
        r_list.append(radii_here)
    
    return r_list, acc_list, worst_weight


def run_screening(x, y, x_test, y_test, weight, lam, weight_change_base):
    n = x.shape[0]
    solutions = {}
    
    print(f'---------- [lambda={lam}] Checking model parameters ----------')
    k = kernel(x,x)
    alpha = train_kernel_dual(k, y, lam, weight)
    solutions[lam] = alpha
    pred, _ = predict(x, x, y, lam, weight, alpha)
    margin = y * pred

    n_pos = np.sum(np.where(y > 0, 1, 0))
    n_neg = n - n_pos
    tp = np.sum((pred > 0) & (y > 0))
    tn = np.sum((pred <= 0) & (y <= 0))
    print(f' Lambda {lam}', end='')
    # print(' TrainingAveLoss %.4f' % np.average(np.maximum(0.0, 1.0 - margin)), end='')    # SVM
    print(' TrainingAveLoss %.4f' % np.average(np.log(1 + np.exp(- margin))), end='')    # Logistic
    print(' TrainingTPR %.4f' % (tp/n_pos), end='')
    print(' TrainingTNR %.4f' % (tn/n_neg), end='')
    print(' TrainingAUC %.4f' % roc_auc_score(y, pred), end='')
    print('')
    
    test_pred, ypred = predict(x, x_test, y, lam, weight, alpha)
    test_score, _ = score(ypred, y_test)
    print(f'{test_score=}')
    print(f'---------- [lambda={lam}] DRCS for weight change by label ----------')
    weight_change = weight_change_base
    new_weight = weight * (weight_change * (y + 1.0) * 0.5 - (y - 1.0) * 0.5)
    weight_dist = np.linalg.norm(new_weight - weight)

    # method
    baselines={"greedy1":remove_greedy1, "greedy2":remove_greedy2, "greedy3":remove_greedy3}
    
    # proposed method
    r_list, acc_list, worst_weight_list = baselines.get(method)(x, y, x_test, y_test, lam, alpha, weight, weight_dist, weight_change_base, pred)

    with open(f'{outdir_csv}/sum_acc_{weight_change_base:.5f}_greedy.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(acc_list)

    # for guarantee
    acc_lowerbound_list, acc_upperbound_list = run_R_Change(x_test, test_pred, y_test, r_list, weight_change_base)

    return acc_list, acc_lowerbound_list, acc_upperbound_list


def relearning(x, y, x_test, y_test, v_current, worst_weight, lam, n, weight_change_base):
    # extract the part whose index is 1
    indices = np.where(v_current == 1)[0]
    x_frac = x[indices]
    y_frac = y[indices]
    weight_frac = worst_weight[indices]
    print(f'number of fraction data: {len(y_frac)}')

    K = kernel(x_frac,x_frac)  
    alpha_frac = train_kernel_dual(K, y_frac, lam, weight_frac)
    
    # evaluation of performance
    pred, y_hat = predict(x_frac, x_test, y_frac, lam, weight_frac, alpha_frac)
    test_score, n_correct = score(y_hat, y_test)
    acc_w, _ = opt_acc(n_correct, y_test, weight_change_base)
    print(f'Relearning[{np.sum(v_current)}/{n}]: test_score = {acc_w}')
    return acc_w


def correct_clf(ypred_pos_index, ypred_neg_index, y_test):
    correct_pos = [i for i in ypred_pos_index if y_test[i] == 1]
    correct_neg = [i for i in ypred_neg_index if y_test[i] == -1]
    incorrect_pos = [i for i in ypred_pos_index if y_test[i] == -1]
    incorrect_neg = [i for i in ypred_neg_index if y_test[i] == 1]
    return correct_pos, correct_neg, incorrect_pos, incorrect_neg


def beta_phix_lowerbound(r, k_test, test_pred):
    return - r * (np.diag(k_test) ** 0.5) + test_pred


def beta_phix_upperbound(r, k_test, test_pred):
    return r * (np.diag(k_test) ** 0.5) + test_pred


def run_R_Change(x_test, test_pred, y_test, r_list, weight_change_base):
    acc_lowerbound_list = []
    acc_upperbound_list = []
    k_test = kernel(x_test, x_test)
    
    for r in r_list:
        t_st = time.time()
        beta_phix_min = beta_phix_lowerbound(r, k_test, test_pred)
        beta_phix_max = beta_phix_upperbound(r, k_test, test_pred)
        
        ypred_pos_index = np.where(beta_phix_min > 0)[0]
        ypred_neg_index = np.where(beta_phix_max < 0)[0]
        # unknown_index = np.where((beta_phix_max > 0) & (beta_phix_min < 0))[0]

        correct_pos, correct_neg, incorrect_pos, incorrect_neg = correct_clf(ypred_pos_index, ypred_neg_index, y_test)
        correct_list = correct_neg + correct_pos
        # incorrect_list = incorrect_neg + incorrect_pos
        # unknown_list = unknown_index
        n_correct = len(correct_list)

        acc_w, w_te = opt_acc(n_correct, y_test, weight_change_base)
        acc_lowerbound_list.append(acc_w)
        # acc_upperbound_list.append(np.dot(w_te, (s+u)) / np.sum(w_te))
        
        t_en = time.time()
        t_run = t_en - t_st
        print(f'TIME(test_optim){t_run=}')
            
    with open(f'{outdir_csv}/sum_acc_bound_{method_name}_{weight_change_base:.5f}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(acc_lowerbound_list)
        writer.writerow(acc_upperbound_list)
    return acc_lowerbound_list, acc_upperbound_list


def main(args):
    x_train, y_train, x_test, y_test, weight, lam, weight_change_base, fold = args
    print(f'=============  Fold {fold}:  =================')
    acc_list, acc_lowerbound_list, acc_upperbound_list = run_screening(x_train, y_train, x_test, y_test, weight, lam, weight_change_base)
    return {'acc': acc_list, 'acc_lb': acc_lowerbound_list, 'acc_ub': acc_upperbound_list}


if __name__ == '__main__':
    from sklearn.datasets import load_svmlight_file
    import concurrent.futures

    if len(sys.argv) != 7 or sys.argv[1] == "":
        sys.stderr.write(f'Usage(1): {sys.argv[0]} [FileName] [OutputDirectory]\n')
        sys.stderr.write(f'Usage(2): {sys.argv[0]} -[PredefinedDataName] [OutputDirectory]\n')
        sys.exit(1)

    fname = sys.argv[1] # filename
    outdir = sys.argv[2] # directory name(dataname)
    way = sys.argv[3]   # select kernel
    method = sys.argv[4]    # select greedy method
    weight_change_base = float(sys.argv[5]) # weight change $a$
    lam = float(sys.argv[6])    # reguralization \lambda
    
    outdir_csv = os.path.join(outdir, way, f'lam_{numform(lam, 4)}')
    os.makedirs(outdir_csv, exist_ok=True)

    trans=True
    if way=="cntk" or way=="resnet_ntk":
        trans=False
    if fname[0] == '-':
        pass
    elif fname == 'cifar10' and way == 'cntk':
        if trans==True:
            transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Lambda(lambda x: x.view(-1))])
        else:
            transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        # download the CIFAR-10 dataset
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        # extract class 1 and class 2
        class_list = [0, 1]
        dataset = [(data, class_list.index(target)) for data, target in trainset if target in class_list]
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True)
        x, y = next(iter(trainloader))
        x = x.numpy()
        y = y.numpy()
        x = x[:1000]
        y = y[:1000]
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        print(f'Number of positive samples: {n_pos}, negative samples: {n_neg}')
        labeler = [0, 1]
        labels = set(y)
        xsub, ysub = extract_by_label(x, y, labeler[0], labeler[1])
    elif fname == 'cifar10' and way == 'rbf':
        pass
    else:
        x, y = load_svmlight_file(fname)
        x = np.asarray(x.todense())
        labeler = sorted(set(y), key=lambda v: -v)
        if len(labeler) != 2:
            sys.stderr.write(f'Error: Not a binary-classification data: "{fname}"\n')
            sys.stderr.write(f'Usage(1): {sys.argv[0]} [FileName]\n')
            sys.stderr.write(f'Usage(2): {sys.argv[0]} -[PredefinedDataName]\n')
            sys.exit(1)

    # Add intercept feature
    if fname == 'cifar10' and way == 'rbf':
        pass
    elif way == "linear" or way == "rbf" or way == "ntk" :
        x = np.hstack((x, np.array([[1.0]] * x.shape[0])))
        # dataset preparation
        n = x.shape[0]
        if y.shape != (n,):
            raise RuntimeError(f"Number of instances not consistent: {n} in `x`, {y.shape} in `y`")
        labels = set(y)
        xsub, ysub = extract_by_label(x, y, labeler[0], labeler[1])
        scaler = StandardScaler()
        xsub = scaler.fit_transform(xsub)

    # select kernel
    kernel_dict={"linear":linear_kernel,"rbf":rbf_kernel,"ntk":cal_ntk,"cntk":cal_cntk}
    kernel = kernel_dict.get(way)

    # K-fold CV
    results = []    
    for iter, random_state in enumerate([42]):
        kf = KFold(shuffle=True, random_state=random_state, n_splits=5)
        method_name = sys.argv[4]
        
        if fname == 'cifar10' and way == 'rbf':
            num_worker = 1
            params = list()
            labeler = [0,1]
            datapath = "./data/cifar-10-features"
            
            for fold in range(num_worker):
                x_train = np.load(os.path.join(datapath, f'z_all_{fold}.npy'))
                y_train = np.load(os.path.join(datapath, f'y_all_{fold}.npy'))
                x_train = np.hstack((x_train, np.array([[1.0]] * x_train.shape[0])))
                labels = set(y_train)
                x_train, y_train = extract_by_label(x_train, y_train, labeler[0], labeler[1])
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = np.load(os.path.join(datapath, f'z_test_all_{fold}.npy'))
                y_test = np.load(os.path.join(datapath, f'y_test_all_{fold}.npy'))
                x_test = np.hstack((x_test, np.array([[1.0]] * x_test.shape[0])))
                labels = set(y_test)
                x_test, y_test = extract_by_label(x_test, y_test, labeler[0], labeler[1])
                scaler = StandardScaler()
                x_test = scaler.fit_transform(x_test)
                
                weight = np.array([1.0] * x_train.shape[0])
                params.append((x_train, y_train, x_test, y_test, weight, lam, weight_change_base, fold))
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker) as executor:
                res = list(executor.map(main, params))
                results.append(res)
        else:
            num_worker = 5
            params = list()
            for fold, (train_index, test_index) in enumerate(kf.split(xsub)):
                x_train, x_test = xsub[train_index], xsub[test_index]
                y_train, y_test = ysub[train_index], ysub[test_index]
                weight = np.array([1.0] * x_train.shape[0])

                params.append((x_train, y_train, x_test, y_test, weight, lam, weight_change_base, fold))

            with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker) as executor:
                res = list(executor.map(main, params))
                results.append(res)
