import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

x = np.array([
    [1, 2],
    [2, 2],
    [2, 3],
    [8, 8],
    [8, 9],
    [25, 30]
])

model = DBSCAN(eps=2, min_samples=2)

clusters = model.fit_predict(x)

print("Cluster Labels:", clusters)

plt.scatter(x[:, 0], x[:, 1], c=clusters)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("DBSCAN Outlier Detection")

plt.show()
