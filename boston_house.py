# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:57:58 2019

@author: Jisan
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

dataset = pd.read_csv('housing.csv',delim_whitespace=True, names=cols)


train_data, test_data = train_test_split(dataset, test_size = .3, random_state = 5)
    
train_x = train_data.drop(columns = ['MEDV'], axis = 1)
train_y = train_data['MEDV']
    
test_x = test_data.drop(columns = ['MEDV'], axis = 1)
test_y = test_data['MEDV'];

train_x = preprocessing.scale(train_x)
test_x = preprocessing.scale(test_x)

model = svm.SVR()

model.fit(train_x, train_y)
predicted = model.predict(test_x)

#print(pd.DataFrame({"Actual": test_y, "Predicted": predicted}).head())


#Manually
d = test_y - predicted
d1 = test_y - np.mean(test_y)
mae = np.mean(np.abs(d))
print("MAE: ", mae)
mse = np.mean(d * d)
print("MSE: ", mse)
rmse = np.sqrt(mse)
print("RMSE: ", rmse)
r_2 = 1 - (sum(d * d) / sum((d1 * d1)))
print("R^2: ", r_2)


#using function

mae1 = mean_absolute_error(test_y, predicted)
print("MAE with sklearn: ", mae1)
mse1 = mean_squared_error(test_y, predicted)
print("MSE with sklearn: ", mse1)
rmse1 = np.sqrt(mse1)
print("RMSE with sklearn: ", rmse1)
r_2_1 = r2_score(test_y, predicted)
print("R^2 with sklearn: ", r_2_1)





