import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time

n = 100000
A = sp.random(n, n, density=10/n, format='csr', data_rvs=np.ones, symmetric=True)
d = np.array(A.sum(axis=1)).flatten()
d_inv = np.where(d>0, 1.0/np.sqrt(d), 0)
M = sp.diags(d_inv) @ A @ sp.diags(d_inv)
L = sp.eye(n) - M

t0 = time.time()
print("Starting LM on M...")
vals, vecs = spla.eigsh(M, k=10, which='LA')
print(f"LM finished in {time.time()-t0:.2f}s")

t0 = time.time()
print("Starting SM on L...")
# This will hang or take very long
vals, vecs = spla.eigsh(L, k=10, which='SM', maxiter=100)
print(f"SM finished in {time.time()-t0:.2f}s")
