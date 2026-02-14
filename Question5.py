import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

kidney_disease_data_frame = pd.read_csv("kidney_disease.csv")
# Read the data set using pd

kidney_disease_data_frame.replace("?", np.nan, inplace =True)
# Replace "?" with NaN,  so the code can interpret missing values

kidney_disease_data_frame.dropna(inplace=True)
# Drop rows with missing values

x = kidney_disease_data_frame.drop("classification", axis=1)
# Create the matrix excluding the "classification" column using the .drop method

x = pd.get_dummies(x)
# Convert the categorical variables into numerical values

y = kidney_disease_data_frame["classification"]
# Create the vector

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)
# Split the data into 70% training and 30% testing
number_of_neighbors = [1,3,5,7,9]
# Create a list of the different k values
accuracy_results = []
# Create a list to store the accuracy results
for k in number_of_neighbors:
    knn_model = KNeighborsClassifier(n_neighbors = k)
# Create the model that iterates through the different k values
    knn_model.fit(x_train, y_train)
# Train the model
    y_pred = knn_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    accuracy_results.append(accuracy)
# Append the accuracy scores to the list

results_table = pd.DataFrame({
    "K":number_of_neighbors,
    "Test Accuracy": accuracy_results
})
# Create a table showing the k value and accuracy results

print(results_table)


highest_accuracy = accuracy_results.index(max(accuracy_results))
best_k_value = number_of_neighbors[highest_accuracy]
# Find the best k value

print("Best k value:", best_k_value)
print("Highest test accuracy:", max(accuracy_results))



'''
Changing the value of k will alter the sensitivity of a model. For example a low k value would result in the model to 
have high variance and be more vulnerable to outliers. On the other hand a high k value results in high bias. This
could cause the model to miss out on identifying key patterns in the data.
 

Very small values of k may result in overfitting, as the model can potentially memorise local patterns. The data would
perform well on the training set, but not the testing set.

A very large k value has the opposite effect as it could potentially miss patterns that exist in the data. This
could result in the model not learning anything about the data. most likely the model would not perform well on the
training set or the testing set.
'''