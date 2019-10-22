# python3
# encoding: utf-8

import numpy as np
from scipy import sparse, array


# INPUT : diag_broadcast - list of diagonals value to broadcast,length equal to 3 or 5; n - integer, band matrix shape.
# OUTPUT : L - 2D np.ndarray, L.shape[0] depends on bandwidth, L.shape[1] = n-1, do not store main diagonal, where all ones;
#                                           add zeros to the right side of rows to handle with changing length of diagonals.
#          U - 2D np.ndarray, U.shape[0] = n, U.shape[1] depends on bandwidth;
#              add zeros to the bottom of columns to handle with changing length of diagonals.
def band_lu(diag_broadcast, n):

    if len(diag_broadcast) == 3:
        a, b, c = diag_broadcast

        L = np.zeros(shape=(1, n - 1))
        U = np.zeros(shape=(n, 2))

        x = np.zeros(shape=(n,))
        y = np.zeros(shape=(n - 1,))
        x[0] = b

        for k in range(1, n):
            y[k - 1] = a / x[k - 1]
            x[k] = b - y[k - 1] * c

        L[0, :] = y
        U[:, 0] = x
        U[:n - 1, 1] = c
        return L, U

    elif len(diag_broadcast) == 5:
        e, d, a, b, c = diag_broadcast

        L = np.zeros(shape=(2, n - 1))
        U = np.zeros(shape=(n, 3))

        x = np.zeros(shape=(n,))
        y = np.zeros(shape=(n,))
        z = np.zeros(shape=(n,))

        x[0] = a
        y[0] = b
        z[1] = d / x[0]

        x[1] = a - y[0] * z[1]
        y[1] = b - z[1] * c
        z[2] = (d - y[0] * e / x[0]) / x[1]

        for k in range(2, n):
            x[k] = a - y[k - 1] * z[k] - e * c / x[k - 2]
            if k < n - 1:
                y[k] = b - z[k] * c
                z[k + 1] = (d - y[k - 1] * e / x[k - 1]) / x[k]

        L[0, :] = z[1:]
        L[1, :n - 2] = e / x[:n - 2]
        U[:, 0] = x
        U[:, 1] = y
        U[:n - 2, 2] = c
        return L, U

    else:
        raise ValueError('Invalid argument')


# INPUT : rectangular matrix A
# OUTPUT: matrices Q - orthogonal and R - upper triangular such that A = QR
def gram_schmidt_qr(A):
    # n > m, r = m
    n, m = A.shape
    Q = np.zeros(shape=(n, m))
    for i in range(m):
        a = A[:, i]
        q = a
        for j in range(i):
            q = q - np.dot(a, Q[:, j]) * Q[:, j]
        q = q / np.linalg.norm(q)
        Q[:, i] = q

    R = np.zeros(shape=(m, m))
    for i in range(m):
        for j in range(m):
            R[i, j] = np.dot(Q[:, i], A[:, j])

    return Q, R


# INPUT : rectangular matrix A
# OUTPUT: matrices Q - orthogonal and R - upper triangular such that A = QR
def modified_gram_schmidt_qr(A):
    # n > m, r = m
    n, m = A.shape
    Q = np.zeros(shape=(n, m))
    for i in range(m):
        a = A[:, i]
        q = a
        for j in range(i):
            q = q - np.dot(q, Q[:, j]) * Q[:, j]
        q = q / np.linalg.norm(q)
        Q[:, i] = q

    R = np.zeros(shape=(m, m))
    for i in range(m):
        for j in range(m):
            R[i, j] = np.dot(Q[:, i], A[:, j])

    return Q, R


# INPUT : rectangular matrix A
# OUTPUT: matrices Q - orthogonal and R - upper triangular such that A=QR
def householder_qr(A):
    # n > m
    n, m = A.shape
    Q = np.identity(n)
    R = A
    for i in range(m):
        x = R[i:, i]
        e = np.zeros_like(x)
        e[0] = 1
        v = x + np.sign(x[0]) * np.linalg.norm(x) * e
        v = v / np.linalg.norm(v)

        H = np.identity(n)
        H_tilda = np.identity(x.shape[0])
        H_tilda = H_tilda - 2 * v[:, np.newaxis] @ v[np.newaxis, :]
        H[i:, i:] = H_tilda

        Q = Q @ H
        R = H @ R
    return Q, R


# INPUT:  G - np.ndarray
# OUTPUT: A - np.ndarray (of size G.shape)
def pagerank_matrix(G):  # 5 pts

    if type(G) != 'scipy.sparse.csr.csr_matrix':
        G = sparse.csr_matrix(G)

    L = array(G.sum(axis=0)).flatten()
    L[L == 0] = 1
    L = 1 / L
    A = G * sparse.spdiags(L.T, 0, *G.shape, format='csr')
    return A


# INPUT:  A - np.ndarray (2D), x0 - np.ndarray (1D), num_iter - integer (positive)
# OUTPUT: x - np.ndarray (of size x0), l - float, res - np.ndarray (of size num_iter + 1 [include initial guess])
def power_method(A, x0, num_iter):
    residuals = []

    x = x0
    x = A.dot(x)
    x /= np.linalg.norm(x)
    Ax = A.dot(x)
    lambd = x.dot(Ax)
    residuals.append(np.linalg.norm(Ax - lambd * x))

    for i in range(num_iter):
        x = A.dot(x)
        x /= np.linalg.norm(x)
        Ax = A.dot(x)
        lambd = x.dot(Ax)
        residuals.append(np.linalg.norm(Ax - lambd * x))

    return x, lambd, residuals


# INPUT:  A - np.ndarray (2D), d - float (from 0.0 to 1.0), x - np.ndarray (1D, size of A.shape[0/1])
# OUTPUT: y - np.ndarray (1D, size of x)
def pagerank_matvec(A, d, x):
    n = A.shape[0]
    dA = d * A
    y = dA.dot(x) + (1 - d) / n * np.sum(x)
    return y
