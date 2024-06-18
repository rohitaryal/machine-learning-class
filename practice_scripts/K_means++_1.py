# Looking at the albo graph looks like
# (from K_means++.py script)
# 4 is appropriate value for n_cluster

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plot

data = load_iris().data
sc = StandardScaler()
x = sc.fit_transform(data)

kmean = KMeans(n_clusters=4, random_state=42, init="k-means++")
kmean.fit(x)
cluster_label = kmean.labels_

plot.plot(cluster_label)
plot.show()