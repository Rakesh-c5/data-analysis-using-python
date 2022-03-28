import pandas as pd
import numpy as np
import copy

df_flights=pd.read_csv("cse.csv")
cat_onehot=df_flights.copy()
cat_onehot=pd.get_dummies(cat_onehot,columns=['m1'],prefix=['m1'])
print(cat_onehot)
