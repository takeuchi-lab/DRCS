import numpy as np

def compute(v, fixed_vector_bottom=False):
    v = np.array(v)
    if len(v.shape) > 1:
        raise RuntimeError(f'Argument must be a vector (Shape of the argument: {v.shape})')

    n = v.shape[0]
    if n <= 1:
        raise RuntimeError('Vector of size 1 or less cannot be accepted')
    
    v = v / np.linalg.norm(v)

    # Main computation
    # 
    # Given a vector [v_1, v_2, v_3, v_4, ... v_n] of norm 1, we would like to compute
    #
    # U = [ v_1  v_2  v_3  v_4  ... v_n  ]
    #     [ a_11 a_12  0    0   ...  0   ]
    #     [ a_21 b_22 b_23  0   ...  0   ]
    #     [ a_31 c_32 c_33 c_34 ...  0   ]
    #     [  :    :    :    :        :   ]
    #     [ a_m1 a_m2 a_m3 a_m4 ... a_mn ],
    # where m = n - 1, so that `U` is orthogonal (any row has norm 1, and
    # any pair of two different rows has inner product 0).
    # [Note] An orthogonal matrix `U` must satisfy UU' = U'U = I.
    #
    # We note that, 
    # - [a_11, a_12] must be orthogonal to [v_1, v_2].
    # - Therefore, in order for the 3rd or latter rows to be orthogonal to the 2nd row,
    #   for any k = {2, 3, ..., m}, [a_k1, a_k2] must be orthogonal to [a_11, a_12].
    #   - This implies that, for any k = {2, 3, ..., m}, [a_k1, a_k2] must be
    #     parallel with [v_1, v_2].
    #
    # The similar holds for the other rows. In general we have the followings:
    # - We can assume that for any h = {1, 2, ..., m} and any k = {h+1, h+2, ..., m},
    #   [a_k1, a_k2, ..., a_{k(h+1)}] must be parallel with [v_1, v_2, ..., v_{h+1}].
    # - Especially, since [a_{(h+1)1}, a_{(h+1)2}, ..., a_{(h+1)(h+1)}] must be parallel
    #   with [v_1, v_2, ..., v_{h+1}], a_{(h+1)(h+2)} is easily determined from the
    #   following constraints:
    #   - there exists a constant t such that "a_{(h+1)j} = t v_j" for any j = 1, 2, ..., h+1,
    #   - Normality constraint: sum_{j=1}^{h+2} a_{(h+1)j}^2 = 1,
    #   - Orthogonality constraint: sum_{j=1}^{h+2} a_{(h+1)j} v_j = 0.
    #   This derives
    #   - S t^2 + a_{(h+1)(h+2)}^2 = 1.
    #   - S t + a_{(h+1)(h+2)} v_{h+2} = 0.
    #     - where S = sum_{j=1}^{h+1} v_j^2
    #   And this is solved as
    #   - a_{(h+1)(h+2)}^2 = S / (S + v_{h+2}^2)
    #   - t^2 = v_{h+2}^2 / {S (S + v_{h+2}^2)}
    #     - Signs of `a_{(h+1)(h+2)}` and `t` must be opposite, but any direction is fine
    #       (In this implementation we set `t` as positive and `a_{(h+1)(h+2)}` as negative)
    #   [Note] In order to assume that `S` is positive, we sort the elements of `v` in
    #          descending order of their absolute values.

    # Sort `v` in descending order of their absolute values
    # `rev_order_v` is computed by:
    # https://stackoverflow.com/questions/28573545/how-to-get-the-index-of-the-sorted-list-for-the-original-list-in-python
    order_v = np.argsort(-np.abs(v))
    rev_order_v = np.empty(n, dtype=np.intp)
    rev_order_v[order_v] = np.arange(n)

    v_reorder = v[order_v]

    # Apply the algorithm above
    # [Note] the 1st row of `a` (the 2nd row of overall matrix) is separately computed
    a = np.zeros((n-1, n))

    S = (v_reorder[0] ** 2) + (v_reorder[1] ** 2)
    a[0, 0] = v_reorder[1] / (S ** 0.5)
    a[0, 1] = -v_reorder[0] / (S ** 0.5)

    for h in range(1, n-1):
        S_new = S + (v_reorder[h+1] ** 2)
        t2 = (v_reorder[h+1] ** 2) / (S * S_new)
        a2 = S / S_new
        a[h, 0:(h+1)] = (t2 ** 0.5) * v_reorder[0:(h+1)]
        a[h, h+1] = -(a2 ** 0.5)
        S = S_new
        
    # Put back the order of `v`, and merge the matrix
    if fixed_vector_bottom:
        return np.vstack((a[:, rev_order_v], v.reshape((1, n))))
    else:
        return np.vstack((v.reshape((1, n)), a[:, rev_order_v]))

if __name__ == '__main__':
    #m = compute([0.48, 0.64, 0.6])
    m = compute([1.0] * 10)
    #m = compute([1.0] * 5)

    np.set_printoptions(precision=4, suppress=True)

    print('m')
    print(m)
    
    print('m m.T')
    print(np.matmul(m, m.T))
    
    print('m.T m')
    print(np.matmul(m.T, m))
