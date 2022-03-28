import pandas as pd
import numpy as np
import copy
df_flights=pd.read_csv("flights.csv")
print(df_flights.info())
cat_df_flights=df_flights.select_dtypes(include=['object'])
print(cat_df_flights['TAIL_NUMBER'])




#LABEL eNCODING
cat_df_flights_lc=cat_df_flights.copy()
cat_df_flights_lc=cat_df_flights_lc['TAIL_NUMBER'].astype('category')
print(cat_df_flights_lc.dtype)

print(df_flights)

cat_df_flights




