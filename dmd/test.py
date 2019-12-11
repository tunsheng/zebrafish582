import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from numpy import dot, multiply, diag, power
from numpy import pi, exp, sin, cos
from dmd import *

# define time and space domains
x = np.linspace(-10, 10, 80)
t = np.linspace(0, 20, 1600)
Xm,Tm = np.meshgrid(x, t)

# create data
D = exp(-power(Xm/2, 2)) * exp(0.8j * Tm)
D += sin(0.9 * Xm) * exp(1j * Tm)
D += cos(1.1 * Xm) * exp(2j * Tm)
D += 0.6 * sin(1.2 * Xm) * exp(3j * Tm)
D += 0.6 * cos(1.3 * Xm) * exp(4j * Tm)
D += 0.2 * sin(2.0 * Xm) * exp(6j * Tm)
D += 0.2 * cos(2.1 * Xm) * exp(8j * Tm)
D += 0.1 * sin(5.7 * Xm) * exp(10j * Tm)
D += 0.1 * cos(5.9 * Xm) * exp(12j * Tm)
D += 0.1 * np.random.randn(*Xm.shape)
D += 0.03 * np.random.randn(*Xm.shape)
D += 5 * exp(-power((Xm+5)/5, 2)) * exp(-power((Tm-5)/5, 2))
D[:800,40:] += 2
D[200:600,50:70] -= 3
D[800:,:40] -= 2
D[1000:1400,10:30] += 3
D[1000:1080,50:70] += 2
D[1160:1240,50:70] += 2
D[1320:1400,50:70] += 2

D = D.T
print(D.shape)
# extract input-output matrices
X = D[:,:-1]
Y = D[:,1:]


# Regular DMD
r = reducedRank(X)
mu, Phi = dmd(X,Y,r)
Psi = timeEvolve(X[:,0], t, mu, Phi)

print("Shape of Phi = ", Phi.shape)
print("Len of Phi = ", len(Phi))
plt.figure()
# plt.imshow(np.abs(Phi), aspect='auto')
plt.plot(Psi[1,:])
plt.plot(Psi[:,1])
plt.show()

D_dmd = dot(Phi, Psi)

# Multi-resolution DMD
print("MRDMD")
nodes = mrdmd(D)
D_mrdmd = [dot(*stitch(nodes, i)) for i in range(7)]



# Visualize results
plt.figure()
plt.suptitle("rDMD individual level")
for i, d in enumerate(D_mrdmd):
	plt.subplot(3,3, i+1)
	plt.imshow(np.abs(d), aspect='auto')
	plt.title("Level " + str(i+1))


plt.figure()
rD_rec = sum(D_mrdmd)
print("rD_rec = ", rD_rec.shape)
plt.subplot(1,3,1)
plt.imshow(np.abs(D), aspect='auto')
plt.title('Original D')
plt.subplot(1,3,2)
plt.imshow(np.abs(D_dmd), aspect='auto')
plt.title("DMD reconstruction of D")
plt.subplot(1,3,3)
plt.imshow(np.abs(rD_rec), aspect='auto')
plt.title("rDMD reconstruction of D")
plt.show()

