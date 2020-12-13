# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 19:35:15 2020

@author: Zeno
"""

# Linear Regression and Multiple Regression Sample

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

#Linear regression 

y = np.array([1000,1250,1300,1350,1600,1700,1800,1900,2100]).reshape(-1,1) #x1 = salary
x = np.array([1,2,3,4,5,6,7,8,9]).reshape(-1,1) # years

# we fit the dataset
linear_sample= LinearRegression()
linear_sample.fit(x,y)

plt.scatter(x, y,color = "red")
plt.xlabel("salary")
plt.ylabel("range of y")
plt.title("prediction of the years effected by salary and over-time")

prediction = linear_sample.predict(x)
print("coef : ",linear_sample.coef_)
print("consant of the linearregression",linear_sample.intercept_)
print("\npredictions : ",prediction)
plt.plot(x,prediction,color = "blue")
plt.show()


