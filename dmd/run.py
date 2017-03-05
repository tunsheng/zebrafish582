import numpy as np
import imageio
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
vid = imageio.get_reader('/home/weyl000/Documents/Assignment/zebrafish582/ZebrafishBrain.mp4')
sizeOfFrame = vid.get_data(0).shape
numFrames = len(vid)
height = sizeOfFrame[0]
width = sizeOfFrame[1]

# First index is frame, then height, width, and RGB
video = np.array([frame for frame in vid ])

# Delete random stuff in image
video[:,:225,:100,:] = 0 # Remove top left
video[:,:220,:125,:] = 0 # Remove top left
video[:,600:,:100,:] = 0
video[:,500:,1000:,:] = 0

DEBUG = False
if DEBUG:
	plt.imshow(video[50])
	plt.show()

###########################################################################################
scale = 10
test = downscale_local_mean(rgb2gray(video[0,:,:,:]), (scale, scale))
height = test.shape[0]
width = test.shape[1]

# row = Time, col = Image
gray_video = np.array([downscale_local_mean(rgb2gray(frame), (scale, scale)).flatten() for frame in video])

# col = Time, Time = Image
gray_video = gray_video.T

if DEBUG:
	print("Shape = ", gray_video.shape)
	im = plt.imshow(gray_video[:, 40].reshape(height, width), cmap='gray')
	plt.show()


DMD = False
MRDMD = True
if DMD:
	# extract input-output matrices
	X = gray_video[:,:-1]
	Y = gray_video[:,1:]
	t = np.linspace(0, numFrames, numFrames)

	# Regular DMD
	r = reducedRank(X)
	print("Rank reduced ", r)
	mu, Phi = dmd(X,Y,r)
	Psi = timeEvolve(X[:,0], t, mu, Phi)

# Multi-resolution DMD
if MRDMD:
	nodes = mrdmd(gray_video)
	D_mrdmd = [dot(*stitch(nodes, i)) for i in range(5)]
	rD_rec = sum(D_mrdmd)
	print("rD_rec = ", rD_rec.shape)
	animateDMD(gray_video, D_mrdmd, numFrames, height, width)
