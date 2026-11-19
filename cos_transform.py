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

#1.6
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

#1.7
Cxy_rec = Cxy_red * Q                  

#y zuerst
Cx_rec = np.zeros((N, N))
for col in range(N):
    I = Cxy_rec[:, col]
    for y in range(N):
        for v in range(N):
            Cx_rec[y, col] += W_inv[y, v] * I[v]

#dann x
img_rec = np.zeros((N, N))
for row in range(N):
    I = Cx_rec[row]
    for x in range(N):
        for u in range(N):
            img_rec[row, x] += W_inv[x, u] * I[u]


# ---------- Fenster 1: ohne Quantisierung ----------
fig1, ax1 = plt.subplots(2, 3, figsize=(14, 9))
fig1.suptitle('1.2 - 1.5: DCT hin und zurück (verlustfrei)')

ax1[0, 0].imshow(img, cmap='gray')
ax1[0, 0].set_title('original image')
ax1[0, 0].set_xlabel('x'); ax1[0, 0].set_ylabel('y')

ax1[0, 1].imshow(Cx, cmap='gray')
ax1[0, 1].set_title('Cx (DCT in x-Richtung)')
ax1[0, 1].set_xlabel('u'); ax1[0, 1].set_ylabel('y')

ax1[0, 2].imshow(Cxy, cmap='gray')
ax1[0, 2].set_title('Cxy (DCT in x und y)')
ax1[0, 2].set_xlabel('u'); ax1[0, 2].set_ylabel('v')

ax1[1, 0].imshow(Cx2, cmap='gray')
ax1[1, 0].set_title('Cx2 (zurück in y)')
ax1[1, 0].set_xlabel('u'); ax1[1, 0].set_ylabel('y')

ax1[1, 1].imshow(img2, cmap='gray')
ax1[1, 1].set_title('img2 (zurück in x)')
ax1[1, 1].set_xlabel('x'); ax1[1, 1].set_ylabel('y')

ax1[1, 2].axis('off')
fig1.tight_layout()

# ---------- Fenster 2: mit Quantisierung ----------
fig2, ax2 = plt.subplots(2, 2, figsize=(10, 9))
fig2.suptitle('1.6 - 1.7: Quantisierung und Rekonstruktion')

ax2[0, 0].imshow(img, cmap='gray')
ax2[0, 0].set_title('original image')
ax2[0, 0].set_xlabel('x'); ax2[0, 0].set_ylabel('y')

ax2[0, 1].imshow(Cxy, cmap='gray')
ax2[0, 1].set_title('Cxy (unquantisiert)')
ax2[0, 1].set_xlabel('u'); ax2[0, 1].set_ylabel('v')

ax2[1, 0].imshow(Cxy_red, cmap='gray')
ax2[1, 0].set_title('Cxy_red (quantisiert)')
ax2[1, 0].set_xlabel('u'); ax2[1, 0].set_ylabel('v')

ax2[1, 1].imshow(img_rec, cmap='gray')
ax2[1, 1].set_title('img_rec (rekonstruiert)')
ax2[1, 1].set_xlabel('x'); ax2[1, 1].set_ylabel('y')

fig2.tight_layout()

plt.show()