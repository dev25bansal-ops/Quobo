import numpy as np
import time
from sklearn.metrics import mutual_info_score
from src.quobo.selection import _discretize
from itertools import combinations

rng = np.random.default_rng(42)
X = rng.normal(size=(400, 50))
y = rng.integers(0, 6, size=400)
Xd = _discretize(X, 16)
n_feat, n_bins, N = 50, 16, 400
n_classes = int(y.max()) + 1

# reference
t0 = time.perf_counter()
I_ref = np.zeros(n_feat)
for j in range(n_feat):
    I_ref[j] = mutual_info_score(Xd[:, j], y)
R_ref = np.zeros((n_feat, n_feat))
for a, b in combinations(range(n_feat), 2):
    m = mutual_info_score(Xd[:, a], Xd[:, b])
    R_ref[a, b] = R_ref[b, a] = m
t_loop = time.perf_counter() - t0

# vectorized contingency via einsum + sklearn on precomputed contingency
O = np.zeros((N, n_feat, n_bins))  # noqa: E741 (one-hot tensor)
for j in range(n_feat):
    O[np.arange(N), j, Xd[:, j]] = 1.0
Yoh = np.zeros((N, n_classes))
Yoh[np.arange(N), y] = 1.0

t0 = time.perf_counter()
Jy = np.einsum("nab,nc->abc", O, Yoh)  # (nf, nbins, nclasses) int counts
J = np.einsum("nai,nbj->abij", O, O)  # (nf, nf, nbins, nbins)
I_v = np.array([mutual_info_score(None, None, contingency=Jy[j]) for j in range(n_feat)])
R_v = np.zeros((n_feat, n_feat))
for a, b in combinations(range(n_feat), 2):
    m = mutual_info_score(None, None, contingency=J[a, b])
    R_v[a, b] = R_v[b, a] = m
t_vec = time.perf_counter() - t0

print(f"loop={t_loop:.3f}s vec={t_vec:.3f}s speedup={t_loop/t_vec:.1f}x")
print("I exact:", np.array_equal(I_ref, I_v))
print("R exact:", np.array_equal(R_ref, R_v))
