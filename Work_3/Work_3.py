import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate random data for clustering
X, y = make_blobs(n_samples=150, n_features=2, centers=4, cluster_std=5, random_state=11)

# Visualisation of initial data
plt.scatter(X[:, 0], X[:, 1])
plt.title("Initial Data")
plt.show()

# List for saving cluster inertia
inertias = []

# Determine the number of clusters from 2 to 10
for k in range(2, 11):
    # Initialising KMeans with a different number of clusters
    kmeans = KMeans(n_clusters=k, random_state=11)
    kmeans.fit(X)
    
    # Add the inertia value to the list
    inertias.append(kmeans.inertia_)

# Building a graph
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), inertias, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.xticks(range(2, 11))
plt.grid(True)
plt.show()
