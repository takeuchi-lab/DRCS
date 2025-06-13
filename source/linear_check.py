from sklearn.datasets import load_svmlight_file
import cvxpy as cp
import numpy as np

def prediction_primal(w, x_test):
    return np.matmul(x_test, w.reshape(-1, 1)).reshape(-1)

def prediction_dual(alpha, y_train, k_train_test):
    return np.matmul(alpha * y_train, k_train_test)

# L2-regularization + L1-loss (hinge loss)
def l2r_l1svm(infile, out_model, out_predict, out_kernel, cost=1.0, bias=-1.0, weight=1.0, kernel='RBF', loss='l1', degree=3, gamma_relative=1.0, coef0=0.0):
    x, y = load_svmlight_file(infile)
    x = np.array(x.todense())
    n, d = x.shape
    gamma = gamma_relative / d # Use 'd' without considering the bias term

    y = np.where(y > 0, 1, -1)

    if isinstance(weight, dict):
        weight = np.array([weight[z] for z in y])
    print('weight', weight)

    if bias >= 0.0:
        x = np.hstack((x, np.ones((n, 1)) * bias))
        d += 1

    if kernel == 'LINEAR':
        # Direct formulation
        z = (y*x.T).T # [y_i x_i]_{i=1}^n
        w = cp.Variable(d)

        if loss == 'l1':
            objfunc = cost * cp.sum(cp.multiply(cp.pos(1.0 - z @ w), weight)) + 0.5 * cp.quad_over_lin(w, 1.0)
            options = {'solver': 'OSQP', 'eps_rel': 1e-12, 'eps_abs': 1e-12, 'max_iter': 1000000}
        elif loss == 'l2':
            objfunc = cost * cp.sum(cp.multiply(cp.pos(1.0 - z @ w) ** 2, weight)) + 0.5 * cp.quad_over_lin(w, 1.0)
            options = {'solver': 'OSQP', 'eps_rel': 1e-12, 'eps_abs': 1e-12, 'max_iter': 1000000}
        elif loss == 'lr': # logistic regression
            objfunc = cost * cp.sum(cp.multiply(cp.logistic(-z @ w), weight)) + 0.5 * cp.quad_over_lin(w, 1.0)
            options = {'solver': 'ECOS', 'abstol': 1e-12, 'reltol': 1e-12, 'max_iters': 1000000}
        else:
            raise RuntimeError(f'Loss "{loss}" not implemented')
        
        prob = cp.Problem(cp.Minimize(objfunc), [])
        result = prob.solve(**options)
        w = w.value
        np.savetxt(out_model+'.primal', w)
        np.savetxt(out_predict+'.primal', prediction_primal(w, x))
        return
    
    # Kernel formulation
    q = np.zeros((n, n))
    k = np.zeros((n, n))
    if kernel == 'RBF':
        for i in range(n):
            for j in range(i, n):
                k[i, j] = np.exp(-gamma*(np.linalg.norm(x[i, :] - x[j, :])**2))
                q[i, j] = y[i] * y[j] * k[i, j]
                q[j, i] = q[i, j]
                k[j, i] = k[i, j]
    elif kernel == 'LINEAR_KERNEL':
        for i in range(n):
            for j in range(i, n):
                k[i, j] = np.sum(x[i, :] * x[j, :])
                q[i, j] = y[i] * y[j] * k[i, j]
                q[j, i] = q[i, j]
                k[j, i] = k[i, j]
    else:
        raise RuntimeError(f'Unknown kernel function: "{kernel}"')

    alpha = cp.Variable(n)
    weighted_cost = cost * weight
    
    if loss == 'l1':
        objfunc = 0.5 * cp.quad_form(alpha, q, assume_PSD=True) - cp.sum(alpha)
        constraint = [alpha >= 0, alpha <= weighted_cost]
        options = {'solver': 'CVXOPT', 'max_iters': 1000000}
    elif loss == 'l2':
        if hasattr(weighted_cost, "__iter__"):
            for i in range(n):
                q[i, i] += 0.5 / weighted_cost[i]
        else:
            for i in range(n):
                q[i, i] += 0.5 / weighted_cost

        objfunc = 0.5 * cp.quad_form(alpha, q, assume_PSD=True) - cp.sum(alpha)
        constraint = [alpha >= 0]
        options = {'solver': 'CVXOPT', 'max_iters': 1000000}
    elif loss == 'lr':
        objfunc = 0.5 * cp.quad_form(alpha, q, assume_PSD=True) - cp.sum(cp.entr(alpha)) - cp.sum(cp.entr(weighted_cost - alpha))
        constraint = [alpha >= (1e-6), alpha <= weighted_cost * (1 - 1e-6)]
        options = {'solver': 'ECOS', 'abstol': 1e-12, 'reltol': 1e-12, 'max_iters': 1000000}
    else:
        raise RuntimeError(f'Loss "{loss}" not implemented')

    prob = cp.Problem(cp.Minimize(objfunc), constraint)
    result = prob.solve(**options)
    print(f'[{kernel}] result: {result}')
    alpha = alpha.value
    np.savetxt(out_model+'.dual', alpha)
    np.savetxt(out_predict+'.dual', prediction_dual(alpha, y, k))

    if kernel == 'LINEAR_KERNEL' and loss == 'l1':
        w = x.T @ (alpha * y)
        np.savetxt(out_model+'.primal', w)
        np.savetxt(out_predict+'.primal', prediction_primal(w, x))
    
    with open(out_kernel, 'w') as f:
        for i in range(n):
            f.write(f'{y[i]} 0:{i+1} ')
            for j in range(n):
                f.write(f'{j+1}:{k[i,j]} ')
            f.write('\n')

import os
OUTDIR='linear_check_py_output'
os.makedirs(OUTDIR, exist_ok=True)
for s in [7]:
    loss_name = {1: 'l2', 3: 'l1', 7: 'lr'}[s]
    for t, tname in {0: "LINEAR", 5: "LINEAR_KERNEL", 2: "RBF"}.items():
        for B in [-1, 1]:
            for W in [1.0, {1: 2, -1: 1}]:
                if B >= 0.0:
                    B_param = f"-B{B}"
                else:
                    B_param = ""
                
                if W == 1:
                    W_param = ""
                else:
                    W_param = "-W" + ".".join([f"{k},{W[k]}" for k in sorted(W.keys())])

                l2r_l1svm("heart_scale", f"{OUTDIR}/model-heart_scale-s{s}-t{t}{B_param}{W_param}.log", f"{OUTDIR}/pred-model-heart_scale-s{s}-t{t}{B_param}{W_param}.log", f'{OUTDIR}/heart_scale-t{t}.kernel', bias=B, kernel=tname, loss=loss_name, weight=W)
