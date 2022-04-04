#4th Apr 2020 DWDM LAB:
#--------------------------------------------------------------------------------------------
#General plot code 1.
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from sklearn.linear_model import LinearRegression
x=[1,2,3,4,5]
y=[10,20,30,40,50]
pl.plot(x,y,linewidth=3,color="red",marker='o',markerfacecolor="blue")
pl.xlabel("x Label")
pl.ylabel("y Label")
pl.title("A GRAPH")
pl.show()
#---------------------------------------------------------------------------
#Linear Regression 2.
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

from sklearn.linear_model import LinearRegression
x=np.array([5,15,25,35,45,55]).reshape(-1,1)
y=np.array([5,20,14,32,22,30])
print(x,y)
model=LinearRegression()
model.fit(x,y)
res=model.score(x,y)
print("score: ",res)
print("Intercept: ",model.intercept_)
print("slope: ",model.coef_)
y_pred=model.predict(x)
print("actual values of y:",y)
print("predicted values of y: ",y_pred)
pl.scatter(x,y,color="black")
pl.plot(x,y_pred,color="blue",linewidth=2,marker="o",markerfacecolor="green")

---------------------------------------------------------------------------------------
