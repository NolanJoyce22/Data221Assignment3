import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


kidney_disease_data_frame = pd.read_csv("kidney_disease.csv")
# Read the data set using pd

kidney_disease_data_frame.replace("?", np.nan, inplace =True)
# Replace "?" with NaN, so the code can interpret missing values

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

knn_model = KNeighborsClassifier(n_neighbors = 5)
# Create the model with the number of neighbors set to 5
knn_model.fit(x_train, y_train)
# Train the model

y_pred= knn_model.predict(x_test)
# Predict the labels of the data

confusion_matrix1 = confusion_matrix(y_test, y_pred)
# Create the confusion matrix
accuracy = accuracy_score(y_test,y_pred)
# Calculate the accuracy
precision = precision_score(y_test, y_pred, pos_label = "ckd")
recall = recall_score(y_test, y_pred, pos_label = "ckd")
# Calculate the recall
f1 = f1_score(y_test, y_pred, pos_label = "ckd")
# Calculate the f1 score

print("Confusion matrix", confusion_matrix1)
print("Accuracy", accuracy)
print("Precision", precision)
print("Recall", recall)
print("F1 Score", f1)

'''
In the context of the kidney disease a true positive occurs when the model correctly predicts a positive test result
for the disease. A true negative represents when the model correctly predicts a negative test result for the disease.
A false positive means the that model incorrectly predicted a positive test, meaning the person does not have the
disease, but the test came back positive. A false negative occurs when the model incorrectly predicts a negative
test result. This means that the person has the disease, but their test came back as negative.

The accuracy score might not be enough alone, as it only calculates the proportion of correct predictions. If the data
contains a majority of either people likely of having kidney disease, or majority of people that are un-likely of having
kidney disease the model could potentially be very accurate, but only because the data contains majority of the
same trait.

The recall metric would be the most important if missing a kidney disease is very serious. Recall measures the power
of the model in the detection of true responses. A high recall would suggest that the model is strong at making
correct identifications. Hence this is why it would be the best to use in this scenario.
'''



