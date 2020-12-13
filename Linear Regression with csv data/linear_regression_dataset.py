# -*- coding: utf-8 -*-
"""
Linear Regression sample with csv file which I created simply
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#reading the data
df = pd.read_csv("linear_regression_dataset.csv",sep = ";") #we seperated columns from ;
x =df.experience.values.reshape(-1,1) #we have to reshape as (-1,1) for not get dimensional problem in visualization
y =df.salary.values # we don't need to reshape it 
#and we fit variables for linear regression model
linear_regression_sample = LinearRegression().fit(x,y)

#Linear Regression formula is y_prediction = b0 + b1*x1
#b0 = constant , b1 = coefficient 
#let's look at these values
print("our constant is : ",linear_regression_sample.intercept_,"\n")
print("our coefficient is : ",linear_regression_sample.coef_,"\n")

#let's look at what is salary when experience is 5 
print(linear_regression_sample.predict([[5]])) #♦it gave 7355 and our real value is 8000

#now we can visualize with linear regression model
y_predict = linear_regression_sample.predict(x)

plt.scatter(x,y,color = "purple")
plt.plot(x,y_predict,color = "brown",label = "linear regression model")
plt.xlabel("experience")
plt.ylabel("salary")
plt.grid(True)
plt.legend()
plt.show()