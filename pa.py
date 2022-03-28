import pandas as pd
import numpy as np
import copy
df_flights=pd.read_csv("flights.csv")
print(df_flights.info())
cat_df_flights=df_flights.select_dtypes(include=['object'])
print(cat_df_flights['UniqueCarrier'])


