import os
import sys
import numpy as np
import cvxpy as cp

sys.path.append(os.path.join(os.path.dirname(__file__), '../python/build/lib.linux-x86_64-cpython-311'))
from liblinear import liblinear, liblinearutil, commonutil

EPSILON = '0.000001' # tolerance threshold of the optimization computations

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
    
    # 重み
    if weight.shape != (n,):
        raise RuntimeError(f'Size of `weight` must be the number of training samples (given shape {weight.shape})')

    return n, d, x, y, weight

def kkt_d2p(alpha, x, y, lam, weight):
    return np.matmul(alpha * y * weight, x) / lam

def train_svm_dual_cvxpy_main(n, wyk_matrix, y, lam, weight):
    # wyk_matrix[i, j] = w[i] * w[j] * y[i] * y[j] * inner_product(x[i, :], x[j, :])

    cp_alpha = cp.Variable(n)
    expr = weight @ cp_alpha - (0.5 / lam) * cp.quad_form(cp_alpha, wyk_matrix, True)

    problem = cp.Problem(cp.Maximize(expr), [0.0 <= cp_alpha, cp_alpha <= 1.0])
    problem.solve(solver=cp.ECOS)
    #problem.solve(warm_start=True)
    #problem.solve(max_iter=100000)

    opt_alpha = cp_alpha.value

    # alphaを制約を満たすよう修正する
    for i in range(n):
        if opt_alpha[i] < 0.0:
            #print(f'alpha[{i}] violating constraint: {opt_alpha[i]} -> 0.0')
            opt_alpha[i] = 0.0
        elif opt_alpha[i] > 1.0:
            #print(f'alpha[{i}] violating constraint: {opt_alpha[i]} -> 1.0')
            opt_alpha[i] = 1.0
    
    # alphaが強制的に修正されたことを踏まえ、再度最適化をする。
    # このために、Coordinate descent（1変数ずつ最適化する）を2ステップだけ適用している。
    #
    # この式の alpha[i] に対する偏微分は
    # weight[i] - lam * wyk_matrix[i, :] @ alpha
    # なので、
    # weight[i] - lam * wyk_matrix[i, :] @ alpha = 0
    # すなわち
    # alpha[i] = (weight[i]/lam - (wyk_matrix[i, :] @ alpha - wyk_matrix[i, i] * alpha[i])) / wyk_matrix[i, i]
    # なる alpha[i] を求める。
    # ・もしこれが0以上1以下なら、alpha[i]はその値にする。
    # ・もしこれが0より小さいなら、alpha[i]は0にする。
    # ・もしこれが1より大きいなら、alpha[i]は1にする。
    # ※s[i, i] == 0 ならこれは直接計算できない。
    #   この場合、weight[i]/lam - (wyk_matrix[i, :] @ alpha - wyk_matrix[i, i] * alpha[i]) が
    #   正のときは alpha[i] = 1、負のときは alpha[i] = 0 とする。
    #   （定義上、s[i, i] >= 0 でないとならないため）
    #   ただし今回の設定の場合、切片特徴が導入されているため、これは生じないはず。
    for k in range(2):
        for i in range(n):
            #if wyk_matrix[i, i] == 0.0:
            #    new_s_alpha = (1.0 if (weight[i]*lam - (wyk_matrix[i, :] @ opt_alpha - wyk_matrix[i, i] * opt_alpha[i])) > 0.0 else 0.0)
            #else:
            new_s_alpha = (weight[i]*lam - (wyk_matrix[i, :] @ opt_alpha - wyk_matrix[i, i] * opt_alpha[i])) / wyk_matrix[i, i]
            new_s_alpha = min(max(new_s_alpha, 0.0), 1.0)
            
            #if new_s_alpha != opt_alpha[i]:
            #    print(f'<k={k}> alpha[{i}]: {opt_alpha[i]} -> {new_s_alpha}')
            opt_alpha[i] = new_s_alpha

    #for i in range(n):
    #    print(cp_alpha.value[i])
    #    print(' ' * 20, opt_alpha[i])
    return opt_alpha

# CVXPYでSVMの学習を行う（双対問題、カーネル行列を指定）
# 花田が書いた svm_ss.py の実装をほぼそのまま持ってきている
def train_kernel_svm_dual_cvxpy(kernel_matrix, y, lam, weight):
    n, _, _, y, weight = prepare_train_svm(None, y, weight)
    if kernel_matrix.shape != (n, n):
        raise RuntimeError(f'`kernel_matrix` must be a matrix of size {n}x{n}')

    wy = weight * y
    wyk_matrix = (wy * (wy * kernel_matrix).T).T
    return train_svm_dual_cvxpy_main(n, wyk_matrix, y, lam, weight)

# CVXPYでSVMの学習を行う（双対問題、線形カーネル限定）
# 花田が書いた svm_ss.py の実装をほぼそのまま持ってきている
def train_linear_svm_dual_cvxpy(x, y, lam, weight):
    n, d, x, y, weight = prepare_train_svm(x, y, weight)

    z_T = (weight * y) * x.T # ブロードキャストの関係でxは転置しておく必要がある
    wyk_matrix = np.matmul(z_T.T, z_T) # z * z.T を意図

    alpha = train_svm_dual_cvxpy_main(n, wyk_matrix, y, lam, weight)

    return {'beta': kkt_d2p(alpha, x, y, lam, weight), 'alpha': alpha}

# CVXPYでSVMの学習を行う（主問題、線形カーネル限定）
# 花田が書いた svm_ss.py の実装をほぼそのまま持ってきている
def train_linear_svm_primal_cvxpy(x, y, lam, weight):
    n, d, x, y, weight = prepare_train_svm(x, y, weight)

    # 主問題を解く
    cp_beta = cp.Variable(d)
    expr = 0.5 * lam * cp.sum_squares(cp_beta) + weight @ cp.pos(1.0 - cp.multiply(y, (x @ cp_beta)))
    problem = cp.Problem(cp.Minimize(expr))
    problem.solve()

    return cp_beta.value

# LIBLINEARでSVMの学習を行う（双対問題、カーネル行列を指定）
# ※「-t 4」がその指定
def train_kernel_dual_liblinear(kernel_matrix, y, lam, weight, problem):
    n, _, _, y, weight = prepare_train_svm(None, y, weight)
    if kernel_matrix.shape != (n, n):
        raise RuntimeError(f'`kernel_matrix` must be a matrix of size {n}x{n}')

    # kernel_matrix を、LIBLINEARでの表現形式に変換
    # カーネル行列を直接与える場合は、カーネル行列の一番左に、（1起点の）事例番号を付けた m×(n+1) 行列と
    # して与えないとならない
    # （m <= n。mがnより小さい可能性があるのは、サポートベクトルのみを残すといった実装に対応するため）
    kernel_matrix_ll = scipy.sparse.csr_matrix(np.hstack((np.arange(1, n+1).reshape((-1, 1)), kernel_matrix)))

    # LIBLINEARでは、正則化の強さを (1/2)||w||_2^2 + C Σ w_i loss_i と設定しているので、
    # C = 1 / lam となる
    c = 1.0 / lam
    model = liblinearutil.train(weight, y, kernel_matrix_ll, f'-s {problem} -t 4 -c {c} -e {EPSILON}')

    # モデルの「w」要素に学習された係数が入る
    # 本来のLIBLINEARなら、ここには主変数しか入り得ないのだが、
    # 花田がLIBLINEARをもとにカーネル対応させるにあたり、ここには
    # 「主問題を解いたなら主変数、双対問題を解いたなら双対変数」が入るようになっている
    alpha_ll, _ = model.get_decfun()
    alpha_ll = np.array(alpha_ll)

    # LIBLINEARの定式化では alpha_i は、花田の定式化に比べて (c*w_i) 倍されているため
    # それを割る
    return alpha_ll / (c * weight)

def train_kernel_svm_dual_liblinear(kernel_matrix, y, lam, weight):
    return train_kernel_dual_liblinear(kernel_matrix, y, lam, weight, 3)

def train_kernel_squaredsvm_dual_liblinear(kernel_matrix, y, lam, weight): # Squared hinge loss
    return train_kernel_dual_liblinear(kernel_matrix, y, lam, weight, 1)

def train_kernel_logistic_dual_liblinear(kernel_matrix, y, lam, weight):
    return train_kernel_dual_liblinear(kernel_matrix, y, lam, weight, 7)


# LIBLINEARでSVMの学習を行う（双対問題、線形カーネル限定）
# ※「-t 5」がその指定
def train_linear_dual_liblinear(x, y, lam, weight, problem):
    n, d, x, y, weight = prepare_train_svm(x, y, weight)

    c = 1.0 / lam
    model = liblinearutil.train(weight, y, scipy.sparse.csr_matrix(x), f'-s {problem} -t 5 -c {c} -e {EPSILON}')
    alpha_ll, _ = model.get_decfun()
    alpha_ll = np.array(alpha_ll)
    alpha = alpha_ll / (c * weight)

    # LIBLINEARの学習結果から予測を行う場合、それ専用の関数が存在するが、
    # その場合にはLIBLINEARの学習結果のオブジェクトそのものが必要なので、それも返している
    return {'beta': kkt_d2p(alpha, x, y, lam, weight), 'alpha': alpha, 'liblinear_model': model}

def train_linear_svm_dual_liblinear(x, y, lam, weight):
    return train_linear_dual_liblinear(x, y, lam, weight, 3)

def train_linear_squaredsvm_dual_liblinear(x, y, lam, weight): # Squared hinge loss
    return train_linear_dual_liblinear(x, y, lam, weight, 1)

def train_linear_logistic_dual_liblinear(x, y, lam, weight):
    return train_linear_dual_liblinear(x, y, lam, weight, 7)

# LIBLINEARでSVMの学習を行う（主問題、線形カーネル限定）
# ※「-t 0」とすると、「内部的には双対問題を解いているが、主変数のみ返す」という設定
#   （そもそも、通常のヒンジ損失＋L2正則化のSVMは、主問題を直接解くのは困難）
def train_linear_primal_liblinear(x, y, lam, weight, problem):
    n, d, x, y, weight = prepare_train_svm(x, y, weight)

    c = 1.0 / lam
    model = liblinearutil.train(weight, y, scipy.sparse.csr_matrix(x), f'-s {problem} -t 0 -c {c} -e {EPSILON}')
    beta, _ = model.get_decfun()

    # LIBLINEARの学習結果から予測を行う場合、それ専用の関数が存在するが、
    # その場合にはLIBLINEARの学習結果のオブジェクトそのものが必要なので、それも返している
    return {'beta': np.array(beta), 'liblinear_model': model}

def train_linear_svm_primal_liblinear(x, y, lam, weight):
    return train_linear_primal_liblinear(x, y, lam, weight, 3)

def train_linear_squaredsvm_primal_liblinear(x, y, lam, weight): # Squared hinge loss
    return train_linear_primal_liblinear(x, y, lam, weight, 1)

def train_linear_logistic_primal_liblinear(x, y, lam, weight):
    return train_linear_primal_liblinear(x, y, lam, weight, 7)

import scipy
import matplotlib.pyplot as plt
import time
from os import path

def measure_time(func, args=[], kwargs={}):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end-start

def main(filename, lam, pos_weight, neg_weight):
    lam = float(lam)
    if lam <= 0:
        raise RuntimeError('lambda must be positive')
    pos_weight = float(pos_weight)
    if pos_weight <= 0:
        raise RuntimeError('pos_weight must be positive')
    neg_weight = float(neg_weight)
    if neg_weight <= 0:
        raise RuntimeError('neg_weight must be positive')

    y_train, x_train = commonutil.svm_read_problem(filename, return_scipy=True)
    y_train = np.array(y_train)
    x_train = np.array(x_train.todense())
    n = y_train.size
    
    weight_train = np.where(np.array(y_train) > 0, pos_weight, neg_weight) # 正例にpos_weight、負例にneg_weightの重み

    # ------------------------------------------------------------
    # カーネルを使わない場合
    # ------------------------------------------------------------

    # 学習計算
    result_linear_dual_cvxpy, time_linear_dual_cvxpy = measure_time(train_linear_svm_dual_cvxpy, (x_train, y_train, lam, weight_train))
    alpha_linear_dual_cvxpy = result_linear_dual_cvxpy['alpha']
    beta_linear_dual_cvxpy = result_linear_dual_cvxpy['beta']
    
    beta_linear_primal_cvxpy, time_linear_primal_cvxpy = measure_time(train_linear_svm_primal_cvxpy, (x_train, y_train, lam, weight_train))

    result_linear_dual_liblinear, time_linear_dual_liblinear = measure_time(train_linear_svm_dual_liblinear, (x_train, y_train, lam, weight_train))
    alpha_linear_dual_liblinear = result_linear_dual_liblinear['alpha']
    beta_linear_dual_liblinear = result_linear_dual_liblinear['beta']
    model_linear_dual_liblinear = result_linear_dual_liblinear['liblinear_model']

    result_linear_primal_liblinear, time_linear_primal_liblinear = measure_time(train_linear_svm_primal_liblinear, (x_train, y_train, lam, weight_train))
    beta_linear_primal_liblinear = result_linear_primal_liblinear['beta']
    model_linear_primal_liblinear = result_linear_primal_liblinear['liblinear_model']

    print('time_linear_dual_cvxpy', time_linear_dual_cvxpy)
    print('time_linear_primal_cvxpy', time_linear_primal_cvxpy)
    print('time_linear_dual_liblinear', time_linear_dual_liblinear)
    print('time_linear_primal_liblinear', time_linear_primal_liblinear)

    # 学習結果がおおむね一致しているかを比較
    order_alpha_linear = np.argsort(alpha_linear_dual_liblinear)
    points_alpha_linear = np.arange(alpha_linear_dual_liblinear.size)
    plt.figure()
    plt.scatter(points_alpha_linear, alpha_linear_dual_liblinear[order_alpha_linear], alpha=0.3, label='alpha [liblinear;dual]')
    plt.scatter(points_alpha_linear, alpha_linear_dual_cvxpy[order_alpha_linear], alpha=0.3, label='alpha [cvxpy;dual]')
    plt.title('Trained alpha (linear SVM)')
    plt.legend()

    outfname = f'{path.basename(filename)}.linearSVM-alpha_{lam}_{pos_weight}_{neg_weight}.pdf'
    plt.savefig(outfname)
    print('Comparison of trained alpha is written to ' + outfname)

    order_beta_linear = np.argsort(beta_linear_dual_liblinear)
    points_beta_linear = np.arange(beta_linear_dual_liblinear.size)
    plt.figure()
    plt.scatter(points_beta_linear, beta_linear_dual_liblinear[order_beta_linear], alpha=0.3, label='beta [liblinear;dual]')
    plt.scatter(points_beta_linear, beta_linear_dual_cvxpy[order_beta_linear], alpha=0.3, label='beta [cvxpy;dual]')
    plt.scatter(points_beta_linear, beta_linear_primal_liblinear[order_beta_linear], alpha=0.3, label='beta [liblinear;primal]')
    plt.scatter(points_beta_linear, beta_linear_primal_cvxpy[order_beta_linear], alpha=0.3, label='beta [cvxpy;primal]')
    plt.title('Trained beta (linear SVM)')
    plt.legend()

    outfname = f'{path.basename(filename)}.linearSVM-beta{lam}_{pos_weight}_{neg_weight}.pdf'
    plt.savefig(outfname)
    print('Comparison of trained beta is written to ' + outfname)

    # ------------------------------------------------------------
    # カーネルを使う場合
    # ------------------------------------------------------------

    # 今回は手計算でRBFカーネルを与える
    kernel_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            val = np.exp(-(np.linalg.norm(x_train[i, :] - x_train[j, :]) ** 2))
            kernel_matrix[i, j] = val
            kernel_matrix[j, i] = val

    # 学習計算
    alpha_kernel_dual_cvxpy, time_kernel_dual_cvxpy = measure_time(train_kernel_svm_dual_cvxpy, (kernel_matrix, y_train, lam, weight_train))
    alpha_kernel_dual_liblinear, time_kernel_dual_liblinear = measure_time(train_kernel_svm_dual_liblinear, (kernel_matrix, y_train, lam, weight_train))
    print('time_kernel_dual_cvxpy', time_kernel_dual_cvxpy)
    print('time_kernel_dual_liblinear', time_kernel_dual_liblinear)

    # 学習結果がおおむね一致しているかを比較
    order_alpha_kernel = np.argsort(alpha_kernel_dual_liblinear)
    points_alpha_kernel = np.arange(alpha_kernel_dual_liblinear.size)
    plt.figure()
    plt.scatter(points_alpha_kernel, alpha_kernel_dual_liblinear[order_alpha_kernel], alpha=0.3, label='alpha [liblinear;dual]')
    plt.scatter(points_alpha_kernel, alpha_kernel_dual_cvxpy[order_alpha_kernel], alpha=0.3, label='alpha [cvxpy;dual]')
    plt.title('Trained alpha (SVM with RBF kernel)')
    plt.legend()

    outfname = f'{path.basename(filename)}.RBFkernelSVM-alpha_{lam}_{pos_weight}_{neg_weight}.pdf'
    plt.savefig(outfname)
    print('Comparison of trained alpha is written to ' + outfname)

    # ------------------------------------------------------------
    # ロジスティック回帰（CVXPYとの比較はせず、SVMとの解の差のみ表示している）
    # ------------------------------------------------------------
    alpha_kernel_dual_liblinear_lr = train_kernel_logistic_dual_liblinear(kernel_matrix, y_train, lam, weight_train)
    print(f'Average of model parameter differences (absolute value) between SVM and logistic regression: {np.sum(np.abs(alpha_kernel_dual_liblinear - alpha_kernel_dual_liblinear_lr)) / alpha_kernel_dual_liblinear_lr.size}')

if __name__ == '__main__':
    import sys

    if len(sys.argv) != 5:
        sys.stderr.write(f'Try several SVM solvers and compare trained coefficients.\n')
        sys.stderr.write(f'\n')
        sys.stderr.write(f'Usage: {sys.argv[0]} [FILENAME] [LAMBDA] [POS_WEIGHT] [NEG_WEIGHT]\n')
        sys.stderr.write(f'[FILENAME]: Data file with LIBSVM format\n')
        sys.stderr.write(f'[LAMBDA]: Strength of L2-regularization (positive real number)\n')
        sys.stderr.write(f'[POS_WEIGHT]: Weights on positive instances (positive real number)\n')
        sys.stderr.write(f'[NEG_WEIGHT]: Weights on negative instances (positive real number)\n')
        sys.exit(-1)
    
    main(*sys.argv[1:5])
