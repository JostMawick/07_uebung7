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

#1.2
img = np.ones((N, N))*255
img[0::7,2:6] = 0
img[1::5,1::5] = 0
img[2:6,0::7] = 0
img[2:5:2,2::3] = 0
img[5,3:5] = 0

Cx = np.zeros((N, N))

for row in range(N):
    I = img[row]                       
    for u in range(N):
        for x in range(N):
            Cx[row, u] += W[u, x] * I[x]


#print(img.round(1))
#print(Cx.round(1))

#1.3
Cxy = np.zeros((N, N))

for col in range(N):
    I = Cx[:, col]                     
    for v in range(N):
        for y in range(N):
            Cxy[v, col] += W[v, y] * I[y]

print(Cxy.round(1))

#1.4
W_inv = np.linalg.inv(W)

Cx2 = np.zeros((N, N))

for col in range(N):
    I = Cxy[:, col]                    
    for y in range(N):
        for v in range(N):
            Cx2[y, col] += W_inv[y, v] * I[v]

#print(Cx2.round(1))

#1.5
img2 = np.zeros((N, N))

for row in range(N):
    I = Cx2[row]                       
    for x in range(N):
        for u in range(N):
            img2[row, x] += W_inv[x, u] * I[u]

#print(img2.round(1))

# base quantization matrix (JPEG standard ITU-T T.81)
Qbase = np.array([[16, 11, 10, 16,  24,  40,  51, 61],
               [12, 12, 14, 19,  26,  58,  60, 55],
               [14, 13, 16, 24,  40,  57,  69, 56],
               [14, 17, 22, 29,  51,  87,  80, 62],
               [18, 22, 37, 56,  68, 109, 103, 77],
               [24, 35, 55, 64,  81, 104, 113, 92],
               [49, 64, 78, 87, 103, 121, 120, 101],
               [72, 92, 95, 98, 112, 100, 103, 99]])

c = 10 # compression quality factor
Q = Qbase*c
Cxy_red = np.round(Cxy / Q).astype(np.int32)

print(Cxy_red)
# nach dem print ist der werte Bereich zwischen -2 und 61. daher brauchen wir nur int8
# C = 32 / 8 = 4 => R=75%

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# --- Obere Reihe: Hinweg ---
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('original image')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')

axes[0, 1].imshow(Cx, cmap='gray')
axes[0, 1].set_title('Cx (DCT in x-Richtung)')
axes[0, 1].set_xlabel('u')
axes[0, 1].set_ylabel('y')

axes[0, 2].imshow(Cxy, cmap='gray')
axes[0, 2].set_title('Cxy (DCT in x- und y-Richtung)')
axes[0, 2].set_xlabel('u')
axes[0, 2].set_ylabel('v')

# --- Untere Reihe: Rückweg ---
axes[1, 0].imshow(img2, cmap='gray')
axes[1, 0].set_title('img2 (Rückwärts in x)')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')

axes[1, 1].imshow(Cx2, cmap='gray')
axes[1, 1].set_title('Cx2 (Rückwärts in y)')
axes[1, 1].set_xlabel('u')
axes[1, 1].set_ylabel('y')

axes[1, 2].axis('off')   # leeres Feld ausblenden

plt.tight_layout()
plt.show()
