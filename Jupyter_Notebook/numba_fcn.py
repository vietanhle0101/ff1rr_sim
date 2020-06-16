from numba import jit
import numpy as np

@jit(nopython=True)
def three_dot(A, B, C):
    return np.dot(A, np.dot(B, C))

@jit(nopython=True)
def four_dot(A, B, C, D):
    return np.dot(A, three_dot(B, C, D))

@jit(nopython=True)
def five_dot(A, B, C, D, E):
    return np.dot(A, four_dot(B, C, D, E))

@jit(nopython=True)
def se_kernel(x, y, v, ls_inv):
    return v**2*np.exp(-0.5*three_dot((x-y).transpose(), ls_inv, (x-y)))

@jit(nopython=True)
def se_kernel_01(x, y, v, ls_inv):
    a = se_kernel(x, y, v, ls_inv)
    b = np.dot(ls_inv, (x-y).transpose())
    return a*b

@jit(nopython=True)
def se_kernel_10(x, y, v, ls_inv):
    a = se_kernel(x, y, v, ls_inv)
    b = -np.dot(ls_inv, (x-y))
    return a*b

@jit(nopython=True)
def se_kernel_11(x, y, v, ls_inv):
    l = len(x)
    a = se_kernel(x, y, v, ls_inv)
    b = np.dot(se_kernel_10(x, y, v, ls_inv).reshape(-1, 1), (x-y).reshape(1,-1))
    return a + b

@jit(nopython=True)
def predict(p, alpha, beta, v, ls_inv, v_noise, Z, N_ind):
    K_star_Z = np.empty(N_ind)
    for i in range(N_ind):
        K_star_Z[i] = se_kernel(p, Z[i, :], v, ls_inv)
    mean = np.dot(K_star_Z, alpha)
    return mean

@jit(nopython=True)
def linearize(p, alpha, beta, v, ls_inv, v_noise, Z, N_ind, n):
    K_star_Z = np.empty(N_ind)
    K10_star_Z = np.empty((n, N_ind))
    for i in range(N_ind):
        K_star_Z[i] = se_kernel(p, Z[i, :], v, ls_inv,)
        K10_star_Z[:, i] = se_kernel_10(p, Z[i, :], v, ls_inv)
    A = np.vstack((K_star_Z.reshape(1,-1), K10_star_Z))
    mx = np.dot(A, alpha)
    return mx
