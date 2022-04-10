#Linear Regression using array 
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
x=np.array([[5,15,25,35,45,55]]).reshape(-1,1)
print(x)
y=np.array([[5,20,14,32,22,38]]).reshape(-1,1)
print(y)

#create an object for LinearRegression
model=LinearRegression()

model.fit(x,y)

res=model.score(x,y)

print("score: ",res)
print("intercept: ",model.intercept_)
print("slope: ",model.coef_)
y_pred=model.predict(x)
print("actual values: ",y)
print("predicted values: ",y_pred)

plt.scatter(x,y,color='black')
plt.plot(x,y_pred)
