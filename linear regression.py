import numpy as np
import matplotlib.pyplot as plt
x=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y=np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
n=np.size(x)
m_x = np.mean(x)
m_y = np.mean(y)
SS_xy = np.sum(y*x) - n*m_y*m_x
SS_xx = np.sum(x*x) - n*m_x*m_x
slope=SS_xy/SS_xx
print("Slope:",slope)
intercept=m_y-(slope*m_x)
print("Intercept:",intercept)
y_pred=slope*x+intercept
print("Actual values of y:",y)
print("Predicted values of y:",y_pred)
plt.scatter(x,y)
plt.plot(x,y_pred)
plt.title("LinearRegression")
plt.xlabel("x-axis")
plt.ylabel("y-axis")