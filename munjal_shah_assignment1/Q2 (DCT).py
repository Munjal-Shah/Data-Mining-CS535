import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import math
import powerlaw


#Dataset 1

dct1 = np.zeros([1000,100])
n1 = 100
count1 = 0

data1 = pd.read_csv('dist1.txt', sep = ' ')
data1 = data1.dropna(axis = 'index')
data1_list = data1.values.tolist()
dataset1 = pd.DataFrame(data1_list)
dataset1.head()

dataset1_transpose1 = dataset1.transpose()
dataset1_transpose1.head()

#DCT code implementation

for i in range(0,1000):
    for j in range(0,100):
        x1 = dataset1_transpose1[i][j]
        if(count1 == 0):
            dct1[i][j] = (math.sqrt((1/n1)) * (math.cos((((2*x1)+1) * count1 * math.pi) / (2*n1))))
        else:
            dct1[i][j] = (math.sqrt((2/n1)) * (math.cos((((2*x1)+1) * count1 * math.pi) / (2*n1)) ))
        count1 += 1

dct_df1 = pd.DataFrame(dct1)
dct_df1.head()

dct_df_transpose1 = dct_df1.transpose()
dct_df_transpose1.head()

dct_df_transpose1['mean'] = dct_df_transpose1.mean(axis=1)
dct_df_transpose1.head()

dct_power1 = dct_df_transpose1['mean'].values
results1 = powerlaw.Fit(dct_power1)

X1 = results1.power_law.alpha
Y1 = results1.power_law.xmin

print(X1)
print(Y1)


#Dataset 2

dct2 = np.zeros([1000,100])
n2 = 100
count2 = 0

data2 = pd.read_csv('dist2.txt', sep = ' ')
data2 = data2.dropna(axis = 'index')
data2_list = data2.values.tolist()
dataset2 = pd.DataFrame(data2_list)
dataset2.head()

dataset2_transpose = dataset2.transpose()
dataset2_transpose.head()

#DCT code implementation

for i in range(0,1000):
    for j in range(0,100):
        x2 = dataset2_transpose[i][j]
        if(count2 == 0):
            dct2[i][j] = (math.sqrt((1/n2)) * (math.cos((((2*x2)+1) * count2 * math.pi) / (2*n2))))
        else:
            dct2[i][j] = (math.sqrt((2/n2)) * (math.cos((((2*x2)+1) * count2 * math.pi) / (2*n2))))
        count2 += 1

dct_df2 = pd.DataFrame(dct2)
dct_df2.head()

dct_df_transpose2 = dct_df2.transpose()
dct_df_transpose2.head()

dct_df_transpose2['mean'] = dct_df_transpose2.mean(axis=1)
dct_df_transpose2.head()

dct_power2 = dct_df_transpose2['mean'].values
results2 = powerlaw.Fit(dct_power2)

X2 = results2.power_law.alpha
Y2 = results2.power_law.xmin

print(X2)
print(Y2)

















