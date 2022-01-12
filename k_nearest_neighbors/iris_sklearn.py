# Solving IRIS databset problem with KNN solution using sklearn library

from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#  Load iris dataset
iris = datasets.load_iris()
iris_X = iris.data # data 
iris_y = iris.target # label

# Split data and labels into train and test lists
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50, shuffle=True)

# Create KNN model
knn = neighbors.KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)

# Print out accuracy score
accuracy = accuracy_score(y_predict, y_test)
print(accuracy)