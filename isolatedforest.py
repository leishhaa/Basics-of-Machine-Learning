import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
data = {
    "Value": [10, 12, 11, 13, 12, 14, 15, 100]
}
df = pd.DataFrame(data)
model = IsolationForest(contamination=0.1, random_state=1)
df["Anomaly"] = model.fit_predict(df[["Value"]])
print(df)
colors = ["red" if x == -1 else "blue" for x in df["Anomaly"]]
plt.scatter(df.index, df["Value"], c=colors)
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Isolation Forest Anomaly Detection")
plt.show()
