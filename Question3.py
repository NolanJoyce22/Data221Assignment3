import pandas as pd
from sklearn.model_selection import train_test_split

kidney_disease_data_frame = pd.read_csv("kidney_disease.csv")
# Read the data set using pd

x = kidney_disease_data_frame.drop("classification", axis=1)
# Create the matrix excluding the "classification" column using the .drop method

y = kidney_disease_data_frame["classification"]
# Create the vector

x_train, x_features, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)
# Split the data into 70% training and 30% testing

'''
You should not train and test a model on the same data, as the model would simply memorize the data. This would result
in overfitting, as well as the model would have very low performance when working with new data.

The purpose of a testing set is to test how well the trained model will perform on new data.
The testing set will either prove or disprove that the model has been trained well based off of its accuracy
with the set.




'''



