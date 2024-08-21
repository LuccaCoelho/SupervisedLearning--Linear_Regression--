import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

# Read CSV data file
data = pd.read_csv("./student-mat.csv", sep=";")

# Get specific data we want from the file
data = data[["G1", "G2", "G3", "freetime", "studytime", "failures", "absences"]]

# What we want to predict
predict = "G3"

# Set our X(Input) and Y (output)
x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

# Create a training module
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y,test_size=0.1)

# Train module 50 times and save the best accuracy to a pickle file
"""
best = 0
for _ in range(50):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y,test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)

    accuracy = linear.score(x_test, y_test)
    print("Acc: ", accuracy)

    if accuracy > best:
        best = accuracy
        with open("student_model.pickle", "wb") as file:
            pickle.dump(linear, file)
"""

# Load saved module
pickle_in = open("student_model.pickle", "rb")
linear = pickle.load(pickle_in)

# Print data in terminal
print("Co: ", linear.coef_)
print("Intercept: ", linear.intercept_)

# Print each prediction made by the module
predictions = np.round(linear.predict(x_test), 0)

for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])

# Plot a simple scatter graph to visualize data correlations
style.use("ggplot")

x = "G1"

pyplot.scatter(data[x], data["G3"])
pyplot.xlabel(x)
pyplot.ylabel("Final Grade")
pyplot.show()
