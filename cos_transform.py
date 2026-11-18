import matplotlib.pyplot as plt
import numpy as np

N = 8

W = np.zeros((N, N))

for u in range(N):
    for x in range(N):
        W[u, x] = np.cos(np.pi * (2 * x + 1) * u / (2 * N))

print(W.round(2))

# Plot: eine Zeile von W pro Subplot
fig, axes = plt.subplots(4, 2, figsize=(10, 9))
for u in range(N):
    ax = axes[u // 2, u % 2]
    ax.stem(range(N), W[u])
    ax.set_title(f"u = {u}")
    ax.set_ylim(-1.1, 1.1)

plt.tight_layout()
plt.show()