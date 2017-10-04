import numpy as np
import Image

k=40   # for changing the number of eigenvalues, simply modify this k

X = np.asarray(Image.open('harvey-saturday-goes7am.jpg').convert('L'))
print(X.shape)
U, s, V = np.linalg.svd(X, full_matrices=True)
m = len(U)
n = len(V)

snot = np.zeros((m,n))
for i in range(0,k):
    snot[i,i] = s[i]

Xnot = np.dot(U,snot)
Xnot = np.dot(Xnot,V)

img = Image.fromarray(((Xnot).astype(np.uint8)))
img.save('out.bmp')
img.show()

#calculate ratio
print(np.linalg.norm(X-Xnot)/np.linalg.norm(X))
