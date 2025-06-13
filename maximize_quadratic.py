from collections import defaultdict
import numpy as np
import find_invsq
import orthonormal_1row as on_1row

import logging
logging.basicConfig()
LOGGER = logging.getLogger(__name__)

def merge_similar_values_(arr, tolerance):
    arr = np.array(arr, dtype=float)
    if len(arr.shape) != 1:
        raise RuntimeError('Only one-dimensional array can be accepted')
    
    s = np.argsort(arr)

    begin_index = 0
    for i in range(len(s)):
        if (i == len(s)-1) or (arr[s[i+1]] - arr[s[i]] > tolerance):
            arr[s[begin_index:(i+1)]] = np.average(arr[s[begin_index:(i+1)]])
            begin_index = i+1
    
    return arr

def validate_input_common_(n, b, c, S, tolerance=1.0e-12):
    b = np.array(b)
    c = np.array(c)
    S = float(S)
    tolerance = float(tolerance)

    if b.shape != (n,):
        raise RuntimeError(f'`b` must be size {n} (given {b.size})')
    if c.shape != (n,):
        raise RuntimeError(f'`c` must be size {n} (given {c.size})')
    
    return b, c, S, tolerance

def validate_input_rawmatrix_(A, b, c, S, tolerance=1.0e-12):
    A = np.array(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise RuntimeError(f'`A` must be a squared matrix (given size {A.shape})')
    n = A.shape[0]

    b, c, S, tolerance = validate_input_common_(n, b, c, S, tolerance)
    return n, A, b, c, S, tolerance

def validate_input_gram_(lam, Z, b, c, S, tolerance=1.0e-12):
    lam = float(lam)
    Z = np.array(Z)

    if lam <= 0:
        raise RuntimeError(f'`lam` must be a positive number (given {lam})')
    
    if len(Z.shape) != 2:
        raise RuntimeError(f'`Z` must be a matrix (given size {Z.shape})')
    n = Z.shape[0]
    
    b, c, S, tolerance = validate_input_common_(n, b, c, S, tolerance)
    return n, lam, Z, b, c, S, tolerance

def validate_input_eigen_(Q, phi, b, c, S, tolerance=1.0e-12):
    Q = np.array(Q)
    phi = np.array(phi)

    if len(Q.shape) != 2 or Q.shape[0] != Q.shape[1]:
        raise RuntimeError(f'`Q` must be a square matrix (given size {Q.shape})')
    n = Q.shape[0]

    if phi.shape != (n,):
        raise RuntimeError(f'`phi` must be size {n} (given {phi.size})')

    b, c, S, tolerance = validate_input_common_(n, b, c, S, tolerance)

    if np.any(phi < -tolerance):
        raise RuntimeError('`A` is not positive semidefinite. (negative eigenvalue found)')

    return n, Q, phi, b, c, S, tolerance

def validate_linear_(eta, tau, n):
    tau = float(tau)

    eta = np.array(eta)
    if eta.shape != (n,):
        raise RuntimeError(f'`eta` must be a vector of size {n} (given: {eta.shape})')
    
    eta_norm = np.linalg.norm(eta)
    return eta / eta_norm, tau / eta_norm

# This library computes the maximization of a quadratic function
# f(w) = w' A w + 2 b'w
# under a convex constraint like ||w - c||_2 <= S or ||w - c||_1 <= S.
# - w, b, c: n-dimensional vector
# - A: n-by-n matrix, positive-semidefinite (i.e. f(w) is convex w.r.t. w)

# Solve f(w) = w' A w + 2 b'w s.t. ||w - c||_2.
def maximize_l2_rawmatrix(A, b, c, S, tolerance=1.0e-12, details=False):
    # ------------------------------------------------------------
    # Check inputs
    # ------------------------------------------------------------
    n, A, b, c, S, tolerance = validate_input_rawmatrix_(A, b, c, S, tolerance)

    # ------------------------------------------------------------
    # Eigendecomposition
    # ------------------------------------------------------------
    A = (A + A.T) / 2
    sigma, Q = np.linalg.eig(A)
    sigma = np.real(sigma)
    Q = np.real(Q)
    phi = sigma
    if phi.size < n:
        phi = np.hstack((phi, np.zeros(n - phi.size)))
    
    if np.any(phi < -tolerance):
        raise RuntimeError('`A` is not positive semidefinite. (negative eigenvalue found)')
    else:
        # Computed eigenvalues may be negative even if we assume `A` is positive semidefinite.
        # So, if they are slightly smaller than zero, we replace them with zeros.
        phi = np.where(phi <= 0, 0, phi)

    return maximize_l2_eigen(Q, phi, b, c, S, tolerance, details, validate=False)

# Solve f(w) = w' A w + 2 b'w s.t. ||w - c||_2 and w'eta = tau.
def maximize_l2_linear_rawmatrix(A, b, c, S, eta, tau, tolerance=1.0e-12, details=False):
    # ------------------------------------------------------------
    # Check inputs
    # ------------------------------------------------------------
    n, A, b, c, S, tolerance = validate_input_rawmatrix_(A, b, c, S, tolerance)
    eta, tau = validate_linear_(eta, tau, n)

    new_radius_squared = S*S - (np.dot(eta, c) - tau)**2
    if new_radius_squared < 0:
        raise RuntimeError('No feasible solution exists under the linear constraint and the radius of the L2-norm constraint')

    # ------------------------------------------------------------
    # Compute linear transformation matrix
    # U[n-1,:] == eta, UU' = U'U = I
    # ------------------------------------------------------------
    U = on_1row.compute(eta, fixed_vector_bottom=True)

    # ------------------------------------------------------------
    # Eigendecomposition
    # ------------------------------------------------------------
    A_transformed = np.matmul(U, np.matmul(A, U.T))

    sigma, Q = np.linalg.eig(A_transformed[0:n-1,0:n-1])
    sigma = np.real(sigma)
    Q = np.real(Q)
    phi = sigma
    if phi.size < n-1:
        phi = np.hstack((phi, np.zeros(n - 1 - phi.size)))
    
    if np.any(phi < -tolerance):
        raise RuntimeError('`A` is not positive semidefinite. (negative eigenvalue found)')
    else:
        # Computed eigenvalues may be negative even if we assume `A` is positive semidefinite.
        # So, if they are slightly smaller than zero, we replace them with zeros.
        phi = np.where(phi <= 0, 0, phi)

    addval = tau * tau * A_transformed[n-1,n-1] + 2 * tau * np.dot(b, eta)
    res = maximize_l2_eigen(Q, phi,
        tau * A_transformed[0:n-1,n-1] + np.matmul(b, U[0:n-1,:].T),
        np.matmul(c, U[0:n-1,:].T),
        new_radius_squared**0.5,
        tolerance, details, validate=False)


    if isinstance(res, list):
        for r in res:
            # print(f'{r=}')
            r['value'] += addval
            r['weights'] = np.matmul(np.hstack((r['weights'], tau)), U)
        return res
    else:
        return res + addval

# Solve f(w) = w' A w + 2 b'w s.t. ||w - c||_2,
# where A = (1/lam) Z Z' (a generalized form of "gram matrix").
# Here, the number of rows of Z must be the same as b and c,
# while the number of columns of Z may not be the same.
# 
# In this case, instead of computing the eigendecomposition of A,
# we have only to compute the singular value decomposition of Z:
# Z = QSV' (Q, V: orthogonal matrices, S: diagonal matrix, same size as Z).
# Then the eigendecomposition of A = (1/lam) Z Z' can be computed as
# A = Q ((1/lam) S S') Q'.
def maximize_l2_gram(lam, Z, b, c, S, tolerance=1.0e-12, details=False):
    n, lam, Z, b, c, S, tolerance = validate_input_gram_(lam, Z, b, c, S, tolerance)

    # Singular value decomposition
    Q, sigma, _ = np.linalg.svd(Z)
    phi = (sigma * sigma) / lam
    if phi.size < n:
        phi = np.hstack((phi, np.zeros(n - phi.size)))
    
    return maximize_l2_eigen(Q, phi, b, c, S, tolerance, details, validate=False)

# Solve f(w) = w' A w + 2 b'w s.t. ||w - c||_2 and w'eta = tau.
def maximize_l2_linear_gram(lam, Z, b, c, S, eta, tau, tolerance=1.0e-12, details=False):
    n, lam, Z, b, c, S, tolerance = validate_input_gram_(lam, Z, b, c, S, tolerance)
    eta, tau = validate_linear_(eta, tau, n)

    new_radius_squared = S*S - (np.dot(eta, c) - tau)**2
    if new_radius_squared < 0:
        raise RuntimeError('No feasible solution exists under the linear constraint and the radius of the L2-norm constraint')

    # ------------------------------------------------------------
    # Compute linear transformation matrix
    # U[n-1,:] == eta, UU' = U'U = I
    # ------------------------------------------------------------
    U = on_1row.compute(eta, fixed_vector_bottom=True)

    # Singular value decomposition
    A_reduced = np.matmul(U[0:n-1,:], Z)
    Q, sigma, _ = np.linalg.svd(A_reduced)
    phi = (sigma * sigma) / lam
    if phi.size < n-1:
        phi = np.hstack((phi, np.zeros(n - 1 - phi.size)))
    
    A_last = np.matmul(U[n-1,:], Z)

    A_transformed_lastrow = np.matmul(A_last, A_reduced.T) / lam
    A_transformed_lastelem = np.dot(A_last, A_last) / lam
    
    addval = tau * tau * A_transformed_lastelem + 2 * tau * np.dot(b, eta)
    res = maximize_l2_eigen(Q, phi,
        tau * A_transformed_lastrow + np.matmul(b, U[0:n-1,:].T),
        np.matmul(c, U[0:n-1,:].T),
        new_radius_squared**0.5,
        tolerance, details, validate=False)
    
    if isinstance(res, list):
        for r in res:
            r['value'] += addval
            r['weights'] = np.matmul(np.hstack((r['weights'], tau)), U)
        return res
    else:
        return res + addval

# Solve f(w) = w' A w + 2 b'w s.t. ||w - c||_2,
# where A = Q diag(phi) Q' (eigendecomposition):
# - Q: orthogonal matrix (QQ' = Q'Q = I)
# - phi: eigenvalues of A (all are nonnegative since A is positive semidefinite)
def maximize_l2_eigen(Q, phi, b, c, S, tolerance=1.0e-12, details=False, validate=True):
    if validate:
        n, Q, phi, b, c, S, tolerance = validate_input_eigen_(Q, phi, b, c, S, tolerance)
    else:
        n = Q.shape[0]
    
    # ------------------------------------------------------------
    # Calculate needed values
    # ------------------------------------------------------------
    phi = np.where(np.abs(phi) <= tolerance, 0.0, phi)
    phi = merge_similar_values_(phi, tolerance)
    #print(f'phi = {phi}')

    # Note: for "v" (vector) and "m" (matrix), in "v * m", "v" multiplies each column of "m"
    xi = (-np.matmul((phi * Q).T, c.reshape(-1, 1)) - np.matmul(Q.T, b.reshape(-1, 1))).reshape(-1)
    xi = np.where(np.abs(xi) <= tolerance, 0.0, xi)
    xi = merge_similar_values_(xi, tolerance)

    # with this setup, the solution is represented as (w, mu) satisfying
    # (diag(phi) - mu I) tau = xi,    ... (1)
    # ||tau||_2 = S,            ... (2)
    # where    tau = Q'(w - c).
    stationary_points = []
    max_m = None
    max_mu = None

    # Case 1: "mu" is the same as one of elements in "phi"
    mu_indices = defaultdict(list)
    for i in range(phi.size):
        mu_indices[phi[i]].append(i)
    
    for mu, match_indices in mu_indices.items():
        # To satisfy the equation (1),
        # - given "mu", let M = { i | phi[i] == mu } be the indices of "phi" whose value is "mu".
        # - Then, "xi[i] == 0" must be satisfied for all "i" in M.
        if np.max(np.abs(xi[match_indices])) > 0:
            LOGGER.info(f'No solution found for mu = {mu}')
            continue
        
        # Initialize "tau"
        set_match_indices = set(match_indices)
        tau = np.zeros(n)
        for i in range(n):
            if i not in set_match_indices:
                tau[i] = xi[i] / (phi[i] - mu)

        # We need to constrain "tau" ||tau||_2 = S,
        # where only a part of values being fixed above.
        # ||tau||_2^2 == S^2
        # ||tau[match_indices]||_2^2 + ||tau[not_match_indices]||_2^2 == S^2
        # ||tau[match_indices]||_2^2 == S^2 - ||tau[not_match_indices]||_2^2
        # This right side is calculated as "sqradius_remainder", which the norm of
        # the elements tau[match_indices] must satisfy.
        sqradius_remainder = S*S - np.dot(tau, tau)
        if sqradius_remainder < 0:
            LOGGER.info(f'"mu" found (mu = {mu}) but radius is zero (S^2={S*S}, ||tau||_2^2={np.dot(tau, tau)})')
            continue

        eta = np.matmul(Q.T, (mu * c + b).reshape((-1, 1))).reshape(-1)
        eta_match_norm = np.linalg.norm(eta[match_indices])
        m_part = np.dot(eta, tau) + eta_match_norm * (sqradius_remainder ** 0.5) # M' in the expression
        m_here = m_part + mu * S * S + np.dot(c, b)

        best_tau = tau.copy()
        if eta_match_norm == 0.0:
            best_tau[match_indices] = eta[match_indices]
            LOGGER.info(f'"mu" found (mu = {mu}) but norm is zero')
        else:
            best_tau[match_indices] = eta[match_indices] * ((sqradius_remainder ** 0.5) / eta_match_norm)
        
        stationary_points.append({
            'type': 's',
            'mu': mu,
            'value': m_here,
            'weights': c + np.matmul(Q, best_tau.reshape(-1, 1)).reshape(-1),
        })
        if max_m is None or m_here > max_m:
            max_m = m_here
            max_mu = mu

    # Case 2: "mu" is not the same as any of elements in "phi"
    mu_stationary = find_invsq.compute(phi, xi*xi, S*S)
    for mu in mu_stationary:
        tau = xi / (phi - mu)
        m_here = np.dot(np.matmul(Q.T, (mu * c + b).reshape((-1, 1))).reshape(-1), tau) + mu * S * S + np.dot(c, b)

        stationary_points.append({
            'type': 'n',
            'mu': mu,
            'value': m_here,
            'weights': c + np.matmul(Q, tau.reshape(-1, 1)).reshape(-1),
        })
        if max_m is None or m_here > max_m:
            max_m = m_here
            max_mu = mu

    if details:
        return stationary_points
    else:
        return max_m

# Maximize the following function
# f(w) = c' A c + 2 b' c + S^2 A_{kk} + 2 d S [A c + b]_k.
# with respect to k = {1, 2, ..., n} and d = {-1, +1}.
#
# We have only to examine this value for all k \in {1, 2, ..., n}.
# ("d" is automatically determined from "[A c + b]_k": choose the same sign.)
#
# If `details` is true, it also return "w", which is calculated as "c + S d e_i".
def maximize_l1_base_(n, c, const_part, S, A, Ac_b, details=False):
    best_value = None
    best_k = None
    best_d = None
    for k in range(n):
        value = const_part + S * S * A[tuple([k] * len(A.shape))] + 2 * S * abs(Ac_b[k])
        if best_value is None or value > best_value:
            best_value = value
            best_k = k
            best_d = (1 if Ac_b[k] >= 0 else -1)
    
    if details:
        best_weight = np.copy(c)
        best_weight[best_k] += S * best_d
        return {
            'value': best_value,
            'weights': best_weight,
        }
    else:
        return best_value

# Solve f(w) = w' A w + 2 b'w s.t. ||w - c||_1 <= S.
# 
# In this case, the maximization of f(w) is known to be achieved at
# one of the "vertex" if the constraint region is a convex polyhedron;
# in this problem "w = c + S d e_k" for d \in {-1, +1}, k \in {1, 2, ..., n},
# e_k = [0, 0, ..., 0, 1, 0, ..., 0] (only the k-th element is 1;
# standard unit vector of the k-th dimension).
#
# So, substituting "w" in "f(w) = w' A w + 2 b'w" by "w = c + S d e_i", we have
# f(w) = c' A c + 2 b' c + S^2 A_{kk} + 2 d S [A c + b]_k.
def maximize_l1_rawmatrix(A, b, c, S, details=False):
    n, A, b, c, S, _ = validate_input_rawmatrix_(A, b, c, S)

    Ac = np.matmul(c, A)
    return maximize_l1_base_(n, c, np.dot(c, Ac) + 2 * np.dot(b, c), S, A, Ac+b, details)

# Solve f(w) = w' A w + 2 b'w s.t. ||w - c||_1 <= S,
# where A = (1/lam) Z Z'.
# Here, the number of rows of Z must be the same as b and c,
# while the number of columns of Z may not be the same.
#
# f(w) = c' A c + 2 b' c + S^2 A_{kk} + 2 d S [A c + b]_k
# = (1/lam) c' Z Z' c + 2 b' c + (1/lam) S^2 [Z Z']_{kk} + 2 d S [(1/lam) Z Z' c + b]_k
# - Here, "[Z Z']_{kk}" is equivalent to ||Z_{k:}||_2^2.
def maximize_l1_gram(lam, Z, b, c, S, details=False):
    n, lam, Z, b, c, S, _ = validate_input_gram_(lam, Z, b, c, S)

    lam_zz = np.linalg.norm(Z, axis=1) / lam
    cZ = np.matmul(c, Z)
    return maximize_l1_base_(n, c, np.dot(cZ, cZ) / lam + 2 * np.dot(b, c), S, lam_zz, np.matmul(cZ, Z.T) / lam + b, details)

if __name__ == '__main__':
    pass
