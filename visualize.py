import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as cm


def animateDMD(original, frames, numFrames, height, width):
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
def showLevel(D_mrdme, time, height, width):
	if (time < D_mrdmd[0].shape[1]) or (time >= 0):
		plt.figure()
		plt.suptitle("rDMD individual level")
		for i, d in enumerate(D_mrdmd):
			plt.subplot(2,3, i+1)
			plt.imshow(np.abs(d[:, time]).reshape(height, width), cmap='gray', aspect='auto')
			plt.title("Level " + str(i+1))
		plt.show()