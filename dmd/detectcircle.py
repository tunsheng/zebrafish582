import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

from visualize import load
from skimage.color import rgb2gray
from skimage.draw import circle_perimeter

from scipy.fftpack import fft, dct

# Return the radius of the circle in a video
def getRadius(video):
	# Get only the circle
	circle = video.copy()
	circle[:,:120,:200,:] = 0
	circle[:, :, 125:,:] = 0
	circle[:, 220:, :,:] = 0
	radii = []

	# Binarize image
	print("Start grayscaling")
	binary_cirlce = [rgb2gray(frames) for frames in circle]
	print("Convert to array")
	binary_cirlce = np.array(binary_cirlce)
	print("Start binarizing")
	binary_cirlce[binary_cirlce < 0.1] = 0
	binary_cirlce[binary_cirlce >= 0.1] = 1


	print("Calculating radius")
	return [radius(image) for image in binary_cirlce]

# Returns the radius of the circle in an image
def radius(image):
	stdy = (max(np.nonzero(image)[0]) - min(np.nonzero(image)[0]))//2
	stdx = (max(np.nonzero(image)[1]) - min(np.nonzero(image)[1]))//2
	return int(round(stdx + stdy)/2)

# Animate the detected circle and return the radius
def visualizeRadius(video, debug=True):
	numFrames = video.shape[0]
	
	# Get only the circle
	circle = video.copy()
	circle[:,:120,:200,:] = 0
	circle[:, :, 125:,:] = 0
	circle[:, 220:, :,:] = 0
	radii = []

	# Binarize image
	print("Start grayscaling")
	binary_cirlce = [rgb2gray(frames) for frames in circle]
	print("Convert to array")
	binary_cirlce = np.array(binary_cirlce)
	print("Start binarizing")
	binary_cirlce[binary_cirlce < 0.1] = 0
	binary_cirlce[binary_cirlce >= 0.1] = 1

	for t in range(numFrames):	
		if debug:
			# Zoom in view
			image = circle[120:220, :125, t]
		else:
			image = circle[:,:, t]

		# Compute circle parameters
		y_pos = int(np.average(np.nonzero(image)[0]))
		x_pos = int(np.average(np.nonzero(image)[1]))
		radius = radius(image)	
		radii.append(radius)

		if debug:
			# Draw circles on images
			cx, cy = circle_perimeter(x_pos, y_pos, radius)
			image[cy, cx] = 0.5
			plt.imshow(image, cmap = 'gray')
			plt.pause(0.01)
	if debug:
		plt.figure()
		plt.show()

	return radii


# Main Function

# Load Video
video = load('/home/weyl000/Documents/Assignment/zebrafish582/ZebrafishBrain.mp4')
radii = getRadius(video)
plt.figure()
plt.subplot(1,3,1)
plt.plot(radii)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Impulse of Zebrafish versus Time")
plt.subplot(1,3,2)
plt.plot(np.abs(fft(radii)))
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.title("Absolute Value of Fast Fourier Transform")
plt.subplot(1,3,3)
plt.plot(dct(radii))
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.title("Discrete Cosine Transform")
plt.show()
