import numpy as np

C = 10
k_c = 3
idx = np.arange(C) * 10
sims = np.random.rand(C, C)
np.fill_diagonal(sims, -1.0)
topk_sub_idx = np.argpartition(sims, -k_c, axis=1)[:, -k_c:]

row_offsets = np.arange(len(idx))[:, None]
topk_sims = sims[row_offsets, topk_sub_idx]

global_rows = np.broadcast_to(idx[row_offsets], topk_sub_idx.shape)
global_cols = idx[topk_sub_idx]

mask = topk_sims > 0.0

print(global_rows[mask].shape)
print(global_cols[mask].shape)
