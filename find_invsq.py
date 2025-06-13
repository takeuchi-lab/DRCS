# Find the root of
# $f(m) := \sum_{i=1}^n (g_i / (m - p_i)^2) = T$
# with respect to $m$, where $g_i > 0$ and $T > 0$.
#
# Note that $f(m)$ is a piecewise convex function with breakpoints
# $\{p_i\}_{i = 1, 2, ..., n}$, and $f(m) \to \infty$ with
# $m \to p_i$ (for any $i = 1, 2, ..., n$).
#
# Moreover, each term of the summation in $f(m)$ must be positive.

import logging
logging.basicConfig()
LOGGER = logging.getLogger(__name__)

import numpy as np
from scipy.optimize import root_scalar

# Find the root of $f(m) := \sum_{i=1}^n (g_i / (m - p_i)^2) = T$
# on condition that
#
# -   $-\infty < m < p_0$ (if `dir == -1`),
# -   $p_{n-1} < m < +\infty$ (if `dir == 1`).
def _compute_for_edge(p, g, T, dir):
    n = p.shape[0]

    if dir != -1 and dir != 1:
        raise RuntimeError('Argument `dir` must be either -1 or 1')
    
    if dir == -1:
        p_edge = p[0]
        g_edge = g[0]
    else:
        p_edge = p[n-1]
        g_edge = g[n-1]

    # Step 1: Find $a$ such that $f(a) > T$ in the search space.
    #
    # -   If `dir == -1`, $a$ can be obtained as "$g_0 / (a - p_0)^2 = T$ and $a < p_0$".
    # -   If `dir == 1`, $a$ can be obtained as "$g_{n-1} / (a - p_{n-1})^2 = T$ and $a > p_{n-1}$".
    # 
    # More specifically,
    # -   If `dir == -1`, $a = p_0 - \sqrt{g_0 / T}$.
    # -   If `dir == 1`, $a = p_{n-1} + \sqrt{g_{n-1} / T}$.
    # To avoid numerical error, in case $a$ computed as above does not satisfy $f(a) > T$,
    # we move $a$ so that $f(a)$ become larger, or more specifically, move $a$ near to
    # $p_0$ (if `dir == -1`) or $p_{n-1}$ (if `dir == 1`).
    a = p_edge + 0.5 * dir * (g_edge / T) ** 0.5
    while True:
        f = np.sum(g / ((a - p) ** 2))
        LOGGER.info(f'Updating a: {a} p_edge: {p_edge} f(a): {f} T: {T}')
        if f > T:
            break
        a = a * 0.75 + p_edge * 0.25

    # Step 2: Find $b$ such that $f(b) < T$ in the search space.
    #
    # Since $f(m) \to +0$ for both $m \to -\infty$ and $m \to +\infty$,
    # -   If `dir == -1`, we have only to take sufficiently small $b$.
    # -   If `dir == 1`, we have only to take sufficiently large $b$.
    b = None
    ext = a - p_edge
    while True:
        b = a + ext
        f = np.sum(g / ((b - p) ** 2))
        if f < T:
            break
        ext *= 2.0

    # Step 3: Solve $f(m) = T$ in the section $[a, b]$
    LOGGER.info(f'CHECK: a={a}, b={b}, f(a)={np.sum(g / ((a - p) ** 2)) - T}, f(b)={np.sum(g / ((b - p) ** 2)) - T}')
    pos = root_scalar(lambda m: np.sum(g / ((m - p) ** 2)) - T, bracket=sorted([a, b]))
    return [pos.root]

# Find the root of $f(m) := \sum_{i=1}^n (g_i / (m - p_i)^2) = T$
# on condition that $p_i < m < p_{i+1}$.
def _compute_for_internal(p, g, T, i):
    n = p.shape[0]
    i = int(i)
    if i < 0 or i >= n-1:
        raise RuntimeError('Argument `i` must be between 0 and (n-1) [n: The number of elements in `p`]')
    
    # Step 1: Find $a$, $b$ such that $f(a) > T$, $f(b) > T$ in the search space.
    # -   Taking $a$ as "$g_i / (a - p_i)^2 = T$, $a > p_i$", and
    # -   taking $b$ as "$g_{i+1} / (b - p_{i+1})^2 = T$, $b < p_{i+1}$
    # assures $f(a) > T$ and $f(b) > T$.
    # 
    # More specifically,
    # -   $a = p_i + \sqrt{g_i / T}$,
    # -   $b = p_{i+1} - \sqrt{g_{i+1} / T}$.
    # To avoid numerical error, in case $a$ computed as above does not satisfy $f(a) > T$
    # we move $a$ so that $f(a)$ become larger, or more specifically, move $a$ near to
    # $p_i$. We do the similar for $b$.
    # 
    # Here, in reality, we can assure that
    # -   $f(a') > T$ for any $p_i < a' \leq a$,
    # -   $f(b') > T$ for any $b \leq b' < p_{i+1}$.
    # So, if $a \geq b$, then no solution of $f(m) = T$ is found in $p_i < m < p_{i+1}$.
    a = p[i] + 0.5 * (g[i] / T) ** 0.5
    while True:
        f = np.sum(g / ((a - p) ** 2))
        if f > T:
            break
        a = a * 0.75 + p[i] * 0.25

    b = p[i+1] - 0.5 * (g[i+1] / T) ** 0.5
    while True:
        f = np.sum(g / ((b - p) ** 2))
        if f > T:
            break
        b = b * 0.75 + p[i+1] * 0.25

    if a >= b:
        return []
    
    # Step 2: Split the section $[a, b]$ so that each section has at most one solution
    # 
    # Since $f(m)$ is strictly convex in $p_i < m < p_{i+1}$, we can find the minimizer
    # $c := \argmin_{m} f(m)$ uniquely. Then,
    # -   If $f(c) > T$, then no solution of $f(m) = T$ is found in $a < m < b$.
    # -   If $f(c) = T$, then $c$ is the unique solution in $a < m < b$.
    # -   If $f(c) < T$, there will be two solutions in each of $[a, c]$ and $[c, b]$.
    #
    # Note that $c$ is computed not by a minimization computation but by $f'(m) = 0$
    # (using the derivative), since $f(m)$ is strictly convex in $p_i < m < p_{i+1}$.
    
    f = lambda c: np.sum(g / ((c - p) ** 2))
    df = lambda c: np.sum(-2 * g / ((c - p) ** 3))
    if df(a) * df(b) > 0:
        return []

    c_sol = root_scalar(df, bracket=sorted([a, b]))
    c = c_sol.root
    optim = f(c)

    if optim > T:
        return []
    elif optim == T:
        return [c]
    
    # Step 3: Solve $f(m) = T$ when $f(c) < T$
    pos_a_c = root_scalar(lambda m: np.sum(g / ((m - p) ** 2)) - T, bracket=[a, c])
    pos_c_b = root_scalar(lambda m: np.sum(g / ((m - p) ** 2)) - T, bracket=[c, b])
    return [pos_a_c.root, pos_c_b.root]

def compute(p, g, T):
    if T <= 0:
        raise RuntimeError(f'Argument `T` must be a positive number (given {T})')

    p = np.array(p)
    g = np.array(g)

    if len(p.shape) != 1 or len(g.shape) != 1:
        raise RuntimeError('Arguments `p` and `g` must be vectors')
    
    if p.shape != g.shape:
        raise RuntimeError('Sizes of `p` and `g` must be the same')

    if p.size == 0:
        raise RuntimeError('Cannot compute for empty vector')

    # Exclude $i \in \{1, 2, \dots, n\}$ such that $g_i = 0$
    if np.any(g < 0.0):
        raise RuntimeError('All elements in `g` must be nonnegative')
    
    idx_g_active = (g != 0.0)
    p = p[idx_g_active]
    g = g[idx_g_active]

    # Sort $\{p_i\}$, and reorder $\{g_i\}$ accordingly
    asort = np.argsort(p)
    p = p[asort]
    g = g[asort]

    n = p.shape[0]
    if n == 0:
        LOGGER.warning('find_invsq.compute: `g_i = 0` for all `i`. No solution found.')
        return []

    result = []

    rs = _compute_for_edge(p, g, T, -1)
    LOGGER.info(f'i={-1} solution={rs}')
    result.extend(rs)
    
    for i in range(n-1):
        rs = _compute_for_internal(p, g, T, i)
        LOGGER.info(f'i={i} solution={rs}')
        result.extend(rs)
    
    rs = _compute_for_edge(p, g, T, 1)
    LOGGER.info(f'i={n-1} solution={rs}')
    result.extend(rs)

    return result

if __name__ == '__main__':
    logging.getLogger(__name__).setLevel(logging.INFO)
    print(compute([1, 3, 4], [3, 4, 6], 50))
    print(compute([1, 3, 4], [3, 4, 6], 30))
