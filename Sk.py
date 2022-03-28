#dealing with categorical data
import pandas as pd
import numpy as np
import copy

from sklearn.preprocessing import LabelEncoder
df_flights=pd.read_csv("cse.csv")
print(df_flights)
#step1:creating an obj of LabelEncoder class

lbobj = LabelEncoder()
#step 2:

#print(df_flights.info())
cat_sklearn_flights=df_flights.copy()
cat_sklearn_flights['name']= lbobj.fit_transform(df_flights['name'])

print(cat_sklearn_flights['name'])
