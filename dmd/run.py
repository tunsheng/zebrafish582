import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from scipy.fftpack import dct

# Required for dmd
from skimage.color import rgb2gray
from skimage.transform import downscale_local_mean
from dmd import *
from visualize import *

# Load Video
# First index is frame, then height, width, and RGB
video = load('/home/weyl000/Documents/Assignment/zebrafish582/ZebrafishBrain.mp4')
numFrames = video.shape[0]

# Get Brain portion
brain = getBrain(video)

# Get Dot portion
circle = video.copy()
circle[:,:120,:200,:] = 0
circle[:, :, 125:,:] = 0
circle[:, 220:, :,:] = 0

DEBUG = False
if DEBUG:
	plt.imshow(brain[50])
	plt.show()

###########################################################################################
scale = 10
test = downscale_local_mean(rgb2gray(video[0,:,:,:]), (scale, scale))
height = test.shape[0]
width = test.shape[1]

# row = Time, col = Image
print("Downscaling images")
gray_video = np.array([downscale_local_mean(rgb2gray(frame), (scale, scale)).flatten() for frame in video])
gray_brain = np.array([downscale_local_mean(rgb2gray(frame), (scale, scale)).flatten() for frame in brain])
gray_circle = np.array([downscale_local_mean(rgb2gray(frame), (scale, scale)).flatten() for frame in circle])

binary_cirlce = gray_circle.copy()
binary_cirlce[binary_cirlce < 0.1] = 0
binary_cirlce[binary_cirlce >= 0.1] = 1

# col = Time, row = Image
print("Transposing matrices")
gray_video = gray_video.T
gray_brain = gray_brain.T
gray_circle = gray_circle.T
binary_cirlce = binary_cirlce.T

# if DEBUG:
# 	print("Shape = ", gray_video.shape)
# 	im = plt.imshow(gray_video[:, 40].reshape(height, width), cmap='gray')
# 	plt.show()

DMD = True
MRDMD = False

# extract input-output matrices
X = gray_brain[:,:-1]
Y = gray_brain[:,1:]
t = np.linspace(0, numFrames, numFrames)

# Regular DMD
r = reducedRank(X)
print("Rank reduced ", r)
mu, Phi = dmd(X,Y)
Psi = timeEvolve(X[:,0], t, mu, Phi)
D_dmd = dot(Phi, Psi)
# animateDMD(gray_video, D_dmd, numFrames, height, width)

# Multi-resolution DMD

# DMD on brain
nodes1 = mrdmd(gray_brain)
D_mrdmd1 = [dot(*stitch(nodes1, i)) for i in range(5)]
rD_rec = sum(D_mrdmd1)
print("rD_rec = ", rD_rec.shape)

# DMD on circle
# nodes2 = mrdmd(binary_cirlce)
# D_mrdmd2 = [dot(*stitch(nodes2, i)) for i in range(5)]

# total = np.abs(D_mrdmd1) + np.abs(D_mrdmd2)
# animateMRDMD(gray_video, total, numFrames, height, width)

# animateMRDMD(gray_video, D_mrdmd1, numFrames, height, width)
# animateMRDMD(gray_video, D_mrdmd2, numFrames, height, width)

animateCompare(gray_video, D_dmd, rD_rec, numFrames, height, width)