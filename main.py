import numpy as np
import imageio
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from scipy.fftpack import dct

# Load Video
vid = imageio.get_reader('ZebrafishBrain.mp4')
sizeOfFrame = vid.get_data(0).shape
numFrames = len(vid)
height = sizeOfFrame[0]
width = sizeOfFrame[1]

# First index is frame, then height, width, and RGB
video = np.array([frame for frame in vid ])

# Delete random stuff in image
video[:,:180,:180,:] = 0
video[:,600:,:100,:] = 0
video[:,500:,1000:,:] = 0

plt.imshow(video[0])
plt.show()

# Do discrete cosine transform
cosTrans = dct(video,axis=0)

# Spectrum of given pixel
singlePixel = cosTrans[:,300,400,0]
plt.plot(singlePixel)
plt.xlabel('Frequency')
plt.show()

## Compute maximum power for each pixel
principalFrequencies = np.max(cosTrans,axis=0)

# Sort pixels by principal frequency
red = principalFrequencies[:,:,0]
vals = np.argsort(red,axis=None)

# Convert values to pixel indices
indices = np.array([(num // width, num % width) for num in vals])

im = video[100, indices[:,0],indices[:,1], :].reshape((height,width,3))
plt.imshow(im)
plt.show()


# Add your analysis code below