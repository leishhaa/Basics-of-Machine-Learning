import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
data = {
    "Age": [18, 19, 20, 35, 36, 37, 50, 51, 52],
    "Income": [20, 22, 25, 50, 52, 55, 80, 82, 85]}
df = pd.DataFrame(data)
x = df[["Age", "Income"]]
linkage_data = linkage(x, method="ward")
dendrogram(linkage_data)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()
model = AgglomerativeClustering(n_clusters=3)
df["Cluster"] = model.fit_predict(x)
print(df)
