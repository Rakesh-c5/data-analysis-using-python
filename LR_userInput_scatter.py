#Linear Regression2 taking input from csv file
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
df=pd.read_csv("DAP/dwdmlab/cse1.csv")
print(df)

x=np.array(df['m1']).reshape(-1,1)
print(x)

y=np.array(df['m2'])
print(y)

model=LinearRegression()
model.fit(x,y)

res=model.score(x,y)
print("score: ",res)
print("intercept: ",model.intercept_)
print("scope: ",model.coef_)

print("actual Values: ",y)
print("predicted values: ",model.predict(x))

plt.scatter(x,y,color='black')
plt.plot(x,model.predict(x))
