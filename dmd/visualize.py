import numpy as np
import imageio
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as cm
from skimage.util import dtype_limits

def load(path):
	vid = imageio.get_reader(path)
	sizeOfFrame = vid.get_data(0).shape
	numFrames = len(vid)
	height = sizeOfFrame[0]
	width = sizeOfFrame[1]

	# First index is frame, then height, width, and RGB
	return np.array([frame for frame in vid ])

def getBrain(video):
	temp = video.copy()
	temp[:,:225,:100,:] = 0 # Remove top left
	temp[:,:220,:125,:] = 0 # Remove top left
	temp[:,600:,:100,:] = 0
	temp[:,500:,1000:,:] = 0
	return temp

def animateDMD(gray_video, D_dmd, numFrames, height, width):
	fig = plt.figure(figsize=(15, 20))
	ax1 = fig.add_subplot(1, 2, 1)
	ax2 = fig.add_subplot(1, 2, 2)

	# Plot DMD
	plt.subplot(1, 2, 1)
	plt.title("DMD")
	ims1 = []
	for n in range(numFrames):
	    im = plt.imshow(np.abs(D_dmd[:, n]).reshape(height, width),
	    				 animated=True, cmap='gray')
	    ims1.append([im])

	# Plot Original
	plt.subplot(1 ,2, 2)
	plt.title("Original")
	ims2 = []
	for n in range(numFrames):
	    im = plt.imshow(gray_video[:, n].reshape(height, width),
	    				 animated=True, cmap='gray')
	    ims2.append([im])
	ani1 = animation.ArtistAnimation(ax1.figure, ims1, interval=100, blit=True,
		                                repeat_delay=100)
	ani2 = animation.ArtistAnimation(ax2.figure, ims2, interval=100, blit=True, 
										repeat_delay=100)
	plt.show()

def animateCompare(gray_video, D_dmd, rD_rec, numFrames, height, width):
	fig = plt.figure(figsize=(15, 20))
	ax1 = fig.add_subplot(1, 3, 1)
	ax2 = fig.add_subplot(1, 3, 2)
	ax3 = fig.add_subplot(1, 3, 3)
	# Plot DMD
	plt.subplot(1, 3, 1)
	plt.title("DMD")
	ims1 = []
	for n in range(numFrames):
	    im = plt.imshow(np.abs(D_dmd[:, n]).reshape(height, width),
	    				 animated=True, cmap='gray')
	    ims1.append([im])

	# Plot MRDMD
	plt.subplot(1, 3, 2)
	plt.title("MRDMD")
	ims2 = []
	for n in range(numFrames):
	    im = plt.imshow(np.abs(rD_rec[:, n]).reshape(height, width),
	    				 animated=True, cmap='gray')
	    ims2.append([im])

	# Plot Original
	plt.subplot(1 ,3, 3)
	plt.title("Original")
	ims3 = []
	for n in range(numFrames):
	    im = plt.imshow(gray_video[:, n].reshape(height, width),
	    				 animated=True, cmap='gray')
	    ims3.append([im])

	ani1 = animation.ArtistAnimation(ax1.figure, ims1, interval=100, blit=True,
		                                repeat_delay=100)
	ani2 = animation.ArtistAnimation(ax2.figure, ims2, interval=100, blit=True, 
										repeat_delay=100)
	ani3 = animation.ArtistAnimation(ax3.figure, ims3, interval=100, blit=True, 
										repeat_delay=100)
	plt.show()


def animateMRDMD(original, frames, numFrames, height, width):
	fig = plt.figure(figsize=(15, 20))
	ax1 = fig.add_subplot(3, 3, 1)
	ax2 = fig.add_subplot(3, 3, 2)
	ax3 = fig.add_subplot(3, 3, 3)
	ax4 = fig.add_subplot(3, 3, 4)
	ax5 = fig.add_subplot(3, 3, 5)
	ax6 = fig.add_subplot(3, 3, 6)
	level1 = frames[0]
	level2 = frames[1]
	level3 = frames[2]
	level4 = frames[3]
	level5 = frames[4]

	# Plot Lvl 1
	plt.subplot(3, 3, 1)
	plt.title("Level 1")
	ims1 = []
	for n in range(numFrames):
	    im = plt.imshow(np.abs(level1[:, n]).reshape(height, width),
	    				 animated=True, cmap='gray')
	    ims1.append([im])


	# Plot Lvl 2
	plt.subplot(3, 3, 2)
	plt.title("Level 2")
	ims2 = []
	for n in range(numFrames):
	    im = plt.imshow(np.abs(level2[:, n]).reshape(height, width),
	    				 animated=True, cmap='gray')
	    ims2.append([im])


	# Plot Lvl 3
	plt.subplot(3, 3, 3)
	plt.title("Level 3")
	ims3 = []
	for n in range(numFrames):
	    im = plt.imshow(np.abs(level3[:, n]).reshape(height, width),
	    				 animated=True, cmap='gray')
	    ims3.append([im])


	# Plot Lvl 4
	plt.subplot(3, 3, 4)
	plt.title("Level 4")
	ims4 = []
	for n in range(numFrames):
	    im = plt.imshow(np.abs(level4[:, n]).reshape(height, width),
	    				 animated=True, cmap='gray')
	    ims4.append([im])


	# Plot Lvl 5
	plt.subplot(3, 3, 5)
	plt.title("Level 5")
	ims5 = []
	for n in range(numFrames):
	    im = plt.imshow(np.abs(level5[:, n]).reshape(height, width),
	    				 animated=True, cmap='gray')
	    ims5.append([im])

	# Plot Original
	plt.subplot(3, 3, 6)
	plt.title("Original")
	ims5 = []
	for n in range(numFrames):
	    im = plt.imshow(original[:, n].reshape(height, width),
	    				 animated=True, cmap='gray')
	    ims5.append([im])
	ani1 = animation.ArtistAnimation(ax1.figure, ims1, interval=100, blit=True,
		                                repeat_delay=100)
	ani2 = animation.ArtistAnimation(ax2.figure, ims2, interval=100, blit=True, 
										repeat_delay=100)
	ani3 = animation.ArtistAnimation(ax3.figure, ims3, interval=100, blit=True,
		                                repeat_delay=100)
	ani4 = animation.ArtistAnimation(ax4.figure, ims4, interval=100, blit=True, 
										repeat_delay=100)
	ani5 = animation.ArtistAnimation(ax5.figure, ims5, interval=100, blit=True, 
										repeat_delay=100)

	plt.show()

# Visualize individual level
def showLevel(D_mrdmd, time, height, width):
	if (time < D_mrdmd[0].shape[1]) or (time >= 0):
		plt.figure()
		plt.suptitle("rDMD individual level")
		for i, d in enumerate(D_mrdmd):
			plt.subplot(2,3, i+1)
			plt.imshow(np.abs(d[:, time]).reshape(height, width), cmap='gray', aspect='auto')
			plt.title("Level " + str(i+1))
		plt.show()



# Invert the intensity of a grayscale image
# Source: Scikit-image 
def invert(image):
    #Invert an image.
    #Substract the image to the maximum value allowed by the dtype maximum.

    if image.dtype == 'bool':
        return ~image
    else:
        return dtype_limits(image, clip_negative=False)[1] - image