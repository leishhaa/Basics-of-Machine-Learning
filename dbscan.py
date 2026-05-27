import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

x, y = make_moons(n_samples=300, noise=0.05, random_state=1)
model = DBSCAN(eps=0.2, min_samples=5)
clusters = model.fit_predict(x)

plt.scatter(x[:, 0], x[:, 1], c=clusters)
plt.title("DBSCAN Clustering")
plt.xlabel("X")
plt.ylabel("Y")

plt.show()
