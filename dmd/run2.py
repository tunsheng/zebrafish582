# This script runs a windowed DMD and do a K-mean clustering

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

# Required for classification
from sklearn.cluster import DBSCAN, KMeans
import time
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics


def dmdWin(gray_brain, numFrames):
	X = gray_brain[:,:-1]
	Y = gray_brain[:,1:]
	t = np.linspace(0, numFrames, numFrames)

	# Regular DMD
	r = reducedRank(X)
	mu, Phi = dmd(X,Y)
	Psi = timeEvolve(X[:,0], t, mu, Phi)
	D_dmd = dot(Phi, Psi)
	return D_dmd, mu


# Load Video
# First index is frame, then height, width, and RGB
video = load('/home/weyl000/Documents/Assignment/zebrafish582/ZebrafishBrain.mp4')
numFrames = video.shape[0]

# Get Brain portion
brain = getBrain(video)

###########################################################################################
scale = 10
test = downscale_local_mean(rgb2gray(video[0,:,:,:]), (scale, scale))
height = test.shape[0]
width = test.shape[1]

# row = Time, col = Image
print("Downscaling images by " + str(scale))
gray_brain = np.array([downscale_local_mean(rgb2gray(frame), (scale, scale)).flatten() for frame in brain])

# col = Time, row = Image
print("Transposing matrices")
gray_brain = gray_brain.T

print("Windowed DMD")
window = 25
overlap = int(0.75*window)
numWindows = floor(1 + (numFrames - window)/overlap)
mu_Win = np.zeros((numWindows, window-1), dtype=complex)
D_dmd, mu = dmdWin(gray_brain[:, 0:window], window)
D_dmdWin = D_dmd
mu_Win[0,:] = mu
for i in range(1, numWindows):
	D_dmd, mu = dmdWin(gray_brain[:, i*overlap:i*overlap+window], window)
	# print("Window DMD shape = ", D_dmd.shape)
	# print("Window DMD eigen shape = ", mu.shape)
	D_dmdWin = np.vstack((D_dmdWin, D_dmd)).astype(complex)
	mu_Win[i,:] = mu
D_dmdWin = np.array(D_dmdWin, dtype=complex)
print("D_dmdWin = ", D_dmdWin.shape)
print("mu_Win = ", mu_Win.shape)
print("numWindows", numWindows)

plt.figure()
plt.plot(np.real(mu_Win[0,:]), np.imag(mu_Win[0,:]), "ro")
plt.plot(np.real(mu_Win[1,:]), np.imag(mu_Win[1,:]), "b+")
plt.plot(np.real(mu_Win[2,:]), np.imag(mu_Win[2,:]), "g*")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.title("Eigenvalues")

radii = np.load("radius.npy")
plt.figure()
plt.plot(radii)
# for i in range(floor(1 + (numFrames - window)/overlap)):
# 	plt.axvline(x=int(i*overlap), color='r')
# 	plt.axvline(x=int(i*overlap+window), color='g')
plt.axvline(x=int(3*overlap), color='r')
plt.axvline(x=int(3*overlap+window), color='g')
plt.xlabel("Time",fontsize=16)
plt.ylabel("Amplitude",fontsize=16)

###########################################################
# DMD_modes as rows

D_dmdWin = np.abs(D_dmdWin)

total_distance = []
for nGroups in range(1, 6):
	k_means = KMeans(init='k-means++', n_clusters=nGroups, n_init=10)
	t0 = time.time()
	k_means.fit(D_dmdWin)
	t_batch = time.time() - t0

	print("Run Time for K-Means = ", t_batch)

	n_clusters = k_means.cluster_centers_
	k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
	k_means_labels = pairwise_distances_argmin(D_dmdWin, k_means_cluster_centers)
	total_distance.append(k_means.inertia_)


plt.figure()
plt.plot(np.linspace(1, 5, 5), total_distance)
plt.xlabel("Number of clusters",fontsize=16)
plt.ylabel("Inertia",fontsize=16)
# plt.show()

k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
k_means.fit(D_dmdWin)
n_clusters = k_means.cluster_centers_
k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
k_means_labels = pairwise_distances_argmin(D_dmdWin, k_means_cluster_centers)
print(k_means_labels[k_means_labels > 0])

plt.figure()
plt.plot(k_means_labels)
plt.xlabel("Modes",fontsize=16)
plt.ylabel("Labels",fontsize=16)
plt.show()
# k_means = KMeans(init='k-means++', n_clusters=2, n_init=10)
# k_means.fit(D_dmdWin)
# n_clusters = k_means.cluster_centers_
# k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
# k_means_labels = pairwise_distances_argmin(D_dmdWin, k_means_cluster_centers)
# print(k_means_labels[k_means_labels > 0])
# print(len(k_means_labels))
# print(len(k_means_labels)//(numWindows*window))
# print(len(mu_WinLabel))
	
###### DBSCAN ########################################
# db = DBSCAN(eps=0.3, min_samples=10).fit(D_dmdWin)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_

# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# print("--DBSCAN--")
# print('Estimated number of clusters: %d' % n_clusters_)
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(D_dmdWin, labels))

# # Black removed and is used for noise instead.
# unique_labels = set(labels)
# colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = 'k'

#     class_member_mask = (labels == k)

#     xy = D_dmdWin[class_member_mask & core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=14)

#     xy = D_dmdWin[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=6)

# plt.title('Estimated number of clusters: %d' % n_clusters_)