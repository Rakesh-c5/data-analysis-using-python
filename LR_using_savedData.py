#Linear Regression with stored data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score
diabetes_X,diabetes_Y=datasets.load_diabetes(return_X_y=True)
diabetes_X=diabetes_X[:,np.newaxis,2]
#print(diabetes_X)
#spliting into training and testing sets
diabetes_X_train=diabetes_X[:-20]
diabetes_X_test=diabetes_X[-20:]
#print(diabetes_X_train)
#print(3*"-------------------------------------------------------")
#print(diabetes_X_test)


#spliting dependent variable
diabetes_Y_train=diabetes_Y[:-20]
diabetes_Y_test=diabetes_Y[-20:]
#print(diabetes_Y_train)
#print(3*"-------------------------------------------------------")
#print(diabetes_Y_test)

#creaing a obj for Linear Regression
regr=linear_model.LinearRegression()
regr.fit(diabetes_X_train,diabetes_Y_train)
diabetes_Y_pred=regr.predict(diabetes_X_test)
print(diabetes_Y_pred)

plt.scatter(diabetes_X_test,diabetes_Y_test)
plt.plot(diabetes_X_test,diabetes_Y_pred,color='black')
print("-------------------------")
print("mean square error: %.2f " %r2_score(diabetes_Y_test,diabetes_Y_pred))
