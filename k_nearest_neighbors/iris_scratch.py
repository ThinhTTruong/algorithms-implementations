# Solving IRIS databset problem with KNN solution made from scratch

from sklearn import datasets
import numpy as np
import math
import operator

# Calculate and return distance between two points
def calculate_distance(p1,p2):
	dimension = len(p1)
	distance = 0

	for i in range(dimension):
		distance += (p1[i] - p2[i])*(p1[i] - p2[i])

	return math.sqrt(distance)

# Find the k nearest neighbors of a given point, and return a list of their labels
def get_k_neighbors(training_X, label_y, point, k):
	distances = []
	neighbors = []

	for i in range(len(training_X)):
		distance = calculate_distance(training_X[i], point)
		distances.append((distance, label_y[i]))

	distances.sort(key=operator.itemgetter(0)) # sort by distance

	for i in range(k):
		neighbors.append(distances[i][1])

	return neighbors

# Find and return the most frequent label from the given label list
def highest_votes(labels):
	labels_count = [0,0,0]
	for label in labels:
		labels_count[label] += 1

	max_count = max(labels_count)
	return labels_count.index(max_count)

# Predict the label of a given point
def predict(training_X, label_y, point, k):
	neighbors_labels = get_k_neighbors(training_X, label_y, point, k)
	return highest_votes(neighbors_labels)

# Return accuracy score between the prediction and the groundtruth
def accuracy_score(predicts, labels):
	total = len(predicts)
	correct_count = 0
	for i in range(total):
		if predicts[i] == labels[i]:
			correct_count += 1
	accuracy = correct_count/total
	return accuracy

# Load iris dataset
iris = datasets.load_iris()
iris_X = iris.data # data 
iris_y = iris.target # label

# Shuffle the data and label list
randIndex = np.arange(iris_X.shape[0])
np.random.shuffle(randIndex)
iris_X = iris_X[randIndex]
iris_y = iris_y[randIndex]

# Split data and labels into train and test lists
X_train = iris_X[:100,:] # 100 training points
X_test = iris_X[100:,:] # 50 testing points
y_train = iris_y[:100] # 100 labels of 100 training points
y_test = iris_y[100:] # 50 labels of 50 testing points

# Get prediction from model and print out the accuracy
k=5
y_predict = []
for p in X_test:
	label = predict(X_train, y_train, p, k)
	y_predict.append(label)

acc = accuracy_score(y_predict, y_test)
print(acc)