import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist

from scipy.spatial.distance import cdist

# Importing data from csv file
data = pd.read_csv('water-treatment.data')

# Adding column names to data
data.columns = ["DATE", "Q_E", "ZN_E", "PH_E", "DBO_E", "DQO_E", "SS_E", "SSV_E", "SED_E", "COND_E", "PH_P", "DBO_P", "SS_P", "SSV_P", "SED_P", "COND_P", "PH_D", "DBO_D", "DQO_D", "SS_D", "SSV_D", "SED_D", "COND_D", "PH_S", "DBO_S", "DQO_S", "SS_S", "SSV_S", "SED_S", "COND_S", "RD_DBO_P", "RD_SS_P", "RD_SED_P", "RD_DBO_S", "RD_DQO_S", "RD_DBO_G", "RD_DQO_G", "RD_SS_G", "RD_SED_G"]

# Removing DATE column
data = data.drop(['DATE'], axis=1)

def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False
		
# Converting all '?' to 'NaN'
data = data[data.applymap(isnumber)]


# Adding missing values using mean of each row
data["Q_E"].fillna(37226.56, inplace=True)
data["ZN_E"].fillna(2.36, inplace=True)
data["PH_E"].fillna(7.81, inplace=True)
data["DBO_E"].fillna(188.71, inplace=True)
data["DQO_E"].fillna(406.89, inplace=True)
data["SS_E"].fillna(227.44, inplace=True)
data["SSV_E"].fillna(61.39, inplace=True)
data["SED_E"].fillna(4.59, inplace=True)
data["COND_E"].fillna(1478.62, inplace=True)

data["PH_P"].fillna(7.83, inplace=True)
data["DBO_P"].fillna(206.20, inplace=True)
data["SS_P"].fillna(253.95, inplace=True)
data["SSV_P"].fillna(60.37, inplace=True)
data["SED_P"].fillna(5.03, inplace=True)
data["COND_P"].fillna(1496.03, inplace=True)

data["PH_D"].fillna(7.81, inplace=True)
data["DBO_D"].fillna(122.34, inplace=True)
data["DQO_D"].fillna(274.04, inplace=True)
data["SS_D"].fillna(94.22, inplace=True)
data["SSV_D"].fillna(72.96, inplace=True)
data["SED_D"].fillna(0.41, inplace=True)
data["COND_D"].fillna(1490.56, inplace=True)

data["PH_S"].fillna(7.70, inplace=True)
data["DBO_S"].fillna(19.98, inplace=True)
data["DQO_S"].fillna(87.29, inplace=True)
data["SS_S"].fillna(22.23, inplace=True)
data["SSV_S"].fillna(80.15, inplace=True)
data["SED_S"].fillna(0.03, inplace=True)
data["COND_S"].fillna(1494.81, inplace=True)

data["RD_DBO_P"].fillna(39.08, inplace=True)
data["RD_SS_P"].fillna(58.51, inplace=True)
data["RD_SED_P"].fillna(90.55, inplace=True)
data["RD_DBO_S"].fillna(83.44, inplace=True)
data["RD_DQO_S"].fillna(67.67, inplace=True)
data["RD_DBO_G"].fillna(89.01, inplace=True)
data["RD_DQO_G"].fillna(77.85, inplace=True)
data["RD_SS_G"].fillna(88.96, inplace=True)
data["RD_SED_G"].fillna(99.08, inplace=True)


#Normalizing data
names = data.columns
scaler = preprocessing.StandardScaler()
scaled_data = scaler.fit_transform(data)
scaled_data = pd.DataFrame(scaled_data, columns=names)

# Applying Elbo method to find number of k in K-Means
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(scaled_data)
    distortions.append(sum(np.min(cdist(scaled_data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / scaled_data.shape[0])
	
	
plt.plot(K, distortions, 'bx-')
plt.xlabel('No. of Clusters')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# From Elbo we are geting k = 2
newk = KMeans(n_clusters=2)
newk.fit(scaled_data)

# Plotting scattered plot for K-Means
y_kmeans = newk.predict(scaled_data)
plt.scatter(scaled_data.iloc[:,0], scaled_data.iloc[:,28], c=y_kmeans, s=30)
plt.show()

# Determining which rows are in which cluster and sending them to output file
labels = newk.labels_
scaled_data['clusters'] = labels
cluster_df = scaled_data['clusters']
cluster_df.to_csv('output.txt', sep=' ')

# Determining components in PCA
pca = PCA().fit(scaled_data)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()

# Getting 90% at components = 15
pca = decomposition.PCA(n_components = 15)
pca_data = pca.fit_transform(scaled_data)
pca1 = pd.DataFrame(pca_data)

# Applying Elbo method to find number of k in K-Means
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(pca1)
    distortions.append(sum(np.min(cdist(pca1, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / pca1.shape[0])


plt.plot(K, distortions, 'bx-')
plt.xlabel('No. of Clusters')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing PCA')
plt.show()

# From Elbo we are geting k = 2
newk = KMeans(n_clusters=2)
newk.fit(pca1)
y_kmeans = newk.predict(pca1)
plt.scatter(pca1.iloc[:,0], pca1.iloc[:,14], c=y_kmeans, s=30)
plt.show()


# Autoencoding

# reduce to 15 features
encoding_dim = 15
input_df = Input(shape=(38,))
encoded = Dense(encoding_dim, activation='relu')(input_df)
decoded = Dense(38, activation='sigmoid')(encoded)
# encoder
autoencoder = Model(input_df, decoded)
# intermediate result
encoder = Model(input_df, encoded)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

autoencoder.fit(scaled_data, scaled_data, epochs=50, batch_size=256, shuffle=True, validation_data=(scaled_data, scaled_data))
encoder_input = Input(shape=(encoding_dim, ))
encoder_out = encoder.predict(scaled_data)

autoencoder = pd.DataFrame(encoder_out)
plt.scatter(autoencoder.iloc[:,1], autoencoder.iloc[:,2], s=30)
plt.show()