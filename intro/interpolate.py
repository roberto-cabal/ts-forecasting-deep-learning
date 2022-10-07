from time_series_data import *
import pandas as pd
import numpy as np

n = 10

ts = get_time_series_data(length=n)

data = pd.DataFrame({'date':pd.date_range(start='2022-01-01', periods=n),'values':ts})

data.loc[np.random.choice(data.index,3),'values'] = 0

print('Zeros maybe are missing')
print(data)
print('\n')

data['values'] = data['values'].replace({0:np.nan})

print('Replace ceros with NaN')
print(data)
print('\n')

data['values'] = data['values'].interpolate()

print('Replace NaNs with interpolation')
print(data)
