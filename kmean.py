import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = {
    "Age": [18, 19, 20, 35, 36, 37, 50, 51, 52],
    "Income": [20, 22, 25, 50, 52, 55, 80, 82, 85]}
df = pd.DataFrame(data)
x = df[["Age", "Income"]]
model = KMeans(n_clusters=3, init="k-means++")
df["Cluster"] = model.fit_predict(x)
print(df)
plt.scatter(df["Age"], df["Income"], c=df["Cluster"])
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("K Means++ Clustering")
plt.show()
