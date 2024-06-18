# K-Means++ Clustering
# Main difference between k-mean and k-mean++ is the
# choosing of random datapoints. In K_means we choose 
# random points randomly and this may introduce a bias
# in the cluster. So this is solved by k-means++ where
# first data point is arbitrarily selected and next
# point is selected by formula z^2 = x^2 + y^2 <= mean
# If this is true then we may select the point else skip

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plot

sc = StandardScaler()

data = load_iris().data
x = sc.fit_transform(data)

wcss = []

for i in range(1,20):
    kmean = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmean.fit(x)
    wcss.append(kmean.inertia_)

plot.figure(figsize=(10,6))
plot.plot(range(1,20), wcss, marker='o', linestyle='--')
plot.title("Albo Graph", fontsize=20)
plot.xticks(range(1,20))
plot.grid(True)
plot.xlabel("Number of cluster", fontsize=15)
plot.ylabel("WCSS", fontsize=15)
plot.show()