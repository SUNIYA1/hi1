# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 21:06:28 2022

"""
#================= SIMPLE LINEAR REGRESSION ON DELVERY_TIME ================

#>>>>>>>>>>>>>>>>>>>>>>>>>> IMPORTING THE DATA SET <<<<<<<<<<<<<<<<<<<<<<<<<<

import pandas as pd 
Delivery=pd.read_csv("D:\\Excel R Assignments\\4. Simple linear Regression\\delivery_time.csv")
Delivery
Delivery.shape       # we get to know dimenssion of data set.
Delivery.head()     # we get first five rows of the columns.
Delivery.info()     # list of the variable names with the data type it is.

Delivery.isnull().sum()   # finding missing values

#>>>>>>>>>>>>>>>>>>>>>>>>> EXPLORATORY DATA ANALYSIS <<<<<<<<<<<<<<<<<<<<<<<

from  scipy.stats import kurtosis
from scipy.stats import skew

Delivery.hist("Sorting Time")       # Moderately positive 
 
kurtosis(Delivery["Sorting Time"],fisher=False)     #1.8346098
skew(Delivery["Sorting Time"])      #0.04368099

Delivery.boxplot("Sorting Time")    # No outliers in the plot.

#>>>>>> Spliting the Dependent and Independent variables from data set <<<<<

X=Delivery["Sorting Time"]
X.ndim
X
import numpy as np
X=np.c_[X]
X.ndim
X
Y=Delivery["Delivery Time"]
Y.ndim

#>>>>>>>>>>>>>>>>> Data Visualization through Scatter Plot <<<<<<<<<<<<<<<<<<

import matplotlib.pyplot as plt
#Delivery.plot.scatter(X="Sorting Time",Y="Delivery Time")
plt.scatter(X,Y,color="red")
plt.show()

Delivery.corr()

#>>>>>>>>>>>>>>>>>>>>>>>>>>>> Fitting the Model <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)

LR.intercept_
LR.coef_

#>>>>>>>>>>>>>>>>>>>>>>>> Predicting the X values <<<<<<<<<<<<<<<<<<<<<<<<<<<

Y_pred=LR.predict(X)
Y_pred

#>>>>>>>>>>>>>>>>>>>>>> Calculating Mean Square Error <<<<<<<<<<<<<<<<<<<<<<<

from sklearn.metrics import mean_squared_error ,r2_score
import numpy as np
MSE=mean_squared_error(Y,Y_pred)
MSE     #7.793311

RMSE=np.sqrt(MSE)           #2.79
print("Root mean squared error of above model is :",RMSE.round(2))

r2 = r2_score(Y,Y_pred)
print("R2: ",r2.round(3))       #0.682
