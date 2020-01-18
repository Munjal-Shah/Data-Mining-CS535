import scipy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#Dataset 1

data1 = pd.read_csv('dist1.txt', sep = ' ')
data1.head()

data1 = data1.dropna(axis = 'index')
data1.head()

datasample1 = data1.sample(10)
datasample1

datasample1['mean'] = datasample1.mean(axis=1)
datasample1['mean']

plt.bar(x, datasample1['mean'], tick_label = x)


#Dataset 2

data2 = pd.read_csv('dist2.txt', sep = ' ')
data2.head()

data2 = data2.dropna(axis = 'index')
data2.head()

datasample2 = data2.sample(10)
datasample2

datasample2['mean'] = datasample2.mean(axis=1)
datasample1['mean']

plt.bar(x, datasample2['mean'], tick_label = x)
