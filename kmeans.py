import numpy as np
import pandas as pd
dataset= pd.read_csv('mall_customers_data.csv')
X = dataset.iloc[:, [3, 4]].values
from sklearn.cluster import KMeans
wcss = []
for i in range(2, 9):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
import matplotlib.pyplot as plt

plt.plot(range(2, 9), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

plt.show()
from sklearn.cluster import KMeans
# Applying KMeans to the dataset
# Choosing 5 clusters based on the elbow method
# The elbow point suggests that 5 clusters is a good choice.
# This is a common practice in clustering to determine the optimal number of clusters.
# The elbow method is a heuristic used to determine the number of clusters in a dataset.
# It involves plotting the within-cluster sum of squares (WCSS) against the number of clusters and looking for a point where the rate of decrease sharply changes, indicating that adding more clusters does not significantly improve the model.
# This point is often referred to as the "elbow" of the curve.
# In this case, the elbow point appears to be at 5 clusters, which is why we will use that number for our KMeans clustering.
             
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)
plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1], s=100,c='red',label='cluster1')
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1], s=100,c='blue',label='cluster2')
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1], s=100,c='orange',label='cluster3')
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1], s=100,c='black',label='cluster4')
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1], s=100,c='purple',label='cluster5')
plt.show()
# Visualizing the clusters

