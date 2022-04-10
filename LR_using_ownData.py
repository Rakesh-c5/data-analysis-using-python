#linear Regression using own data
#demo
#Linear Regression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score

#print(diabetes_X)
#spliting into training and testing sets

df=pd.read_csv("DAP/dwdmlab/cse1.csv")
print(df.info())
diabetes_X=np.array([df['m1']]).reshape((-1,1))
diabetes_Y=np.array([df['m2']])

diabetes_X_train=np.array([df['m1'][:15]]).reshape((-1,1))
diabetes_X_test=np.array([df['m1'][15:]]).reshape((-1,1))
#print(diabetes_X_train)
#print(3*"-------------------------------------------------------")
#print(diabetes_X_test)


#spliting dependent variable
diabetes_Y_train=np.array([df['m2'][:15]]).reshape((-1,1))
diabetes_Y_test=np.array([df['m2'][15:]]).reshape((-1,1))
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
print("r2 square value: %.2f " %r2_score(diabetes_Y_test,diabetes_Y_pred))
print("mean square error:%.2f" %mean_squared_error(diabetes_Y_test,diabetes_Y_pred))

