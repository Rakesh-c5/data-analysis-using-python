import pandas as pd
import numpy as np

import copy
from sklearn.preprocessing import LabelBinarizer

basic=pd.read_csv("cse.csv")
print("\n",basic)
print("\n",basic.info())
print("\n")
cat_df_flights=basic.select_dtypes(include=['object'])
print(cat_df_flights['gender'])
cat_df_flights=basic.select_dtypes(include=['int64'])
print(cat_df_flights['rno'])

cat_onehot=basic.copy()
lb_obj=LabelBinarizer()
lb_results=lb_obj.fit_transform(basic['gender'])
print(lb_results)


