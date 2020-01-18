import pandas as pd
import numpy as np
import powerlaw

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA

sc = StandardScaler()

ica2 = FastICA(n_components = 2)
ica80 = FastICA(n_components = 80)

#Dataset 1

data1 = pd.read_csv('dist1.txt', sep = ' ')
data1.head()

data1 = data1.dropna(axis = 'index')
dataset1 = data1.values
data1_std = sc.fit_transform(dataset1)

data1_std_ica2 = ica2.fit_transform(data1_std)
data1_std_ica80 = ica80.fit_transform(data1_std)


data1_std_ica2
data1_std_ica80

plt.scatter(data1_std_ica2[:,0], data1_std_ica2[:,1])

dataset1frame80 = pd.DataFrame(data1_std_ica80)
dataset1frame80.head()

dataset1frame80['mean'] = dataset1frame80.mean(axis=1)
dataset1frame80.head()

ica_power1 = dataset1frame80['mean'].values
results1 = powerlaw.Fit(ica_power1)

X1 = results1.power_law.alpha
Y1 = results1.power_law.xmin

print(X1)
print(Y1)


#Dataset 2

data2 = pd.read_csv('dist1.txt', sep = ' ')
data2.head()

data2 = data2.dropna(axis = 'index')
dataset2 = data2.values
data2_std = sc.fit_transform(dataset2)

data2_std_ica2 = ica2.fit_transform(data2_std)
data2_std_ica80 = ica80.fit_transform(data2_std)

data2_std_ica2
data2_std_ica80

plt.scatter(data2_std_ica[:,0], data2_std_ica[:,1])

dataset2frame80 = pd.DataFrame(data2_std_ica80)
dataset2frame80.head()

dataset2frame80['mean'] = dataset2frame80.mean(axis=1)
dataset2frame80.head()

ica_power2 = dataset2frame80['mean'].values
results2 = powerlaw.Fit(ica_power2)

X2 = results2.power_law.alpha
Y2 = results2.power_law.xmin

print(X2)
print(Y2)