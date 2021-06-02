
# importing library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
data_set=pd.read_csv("Book2.csv")

# extracting dependent and independent variable
x=data_set.iloc[:,2:5].values
y=data_set.iloc[:,1].values

#splitting data set  into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# Fitting multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the Test set result; 
y_pred= regressor.predict(x_test)

#print(y_pred)


print('Train Score: ', regressor.score(x_train, y_train)) 
print('Test Score: ', regressor.score(x_test, y_test))


print('slope of regression line', regressor.intercept_)
print(' coefficient of the line', regressor.coef_) 