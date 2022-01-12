# Application of K-Means in image compression

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy

# Read image and return 3-dimensional array
img = plt.imread("tree.jpg")

# Get width and height from image shape
width = img.shape[0]
height = img.shape[1]

# Turn image shape to 2D
img = img.reshape(width*height,3)

# Create K-Mean model and fit the given data
kmeans = KMeans(n_clusters=5).fit(img)

# Get labels list and cluster points
labels = kmeans.predict(img)
clusters = kmeans.cluster_centers_

# Create a compressed image
img2 = numpy.zeros_like(img)

for i in range(len(img2)):
	img2[i] = clusters[labels[i]]

img2 = img2.reshape(width,height,3)

plt.imshow(img2)
plt.show()