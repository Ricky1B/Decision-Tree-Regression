# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:51:37 2019

@author: hp
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Fitting the decision tree regression to dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#Prediction for 6.5 year of experience
y_pred=regressor.predict(np.array([[6.5]]))

#Visualising the results(as dtr divides x into intervals for 1d data we need to have highresolution of graph to ibserve the intervals)
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.show()