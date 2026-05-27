import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [25, 35, 45, 55, 65, 75, 85, 95],
    "Pass": [0, 0, 0, 0, 1, 1, 1, 1]}
df = pd.DataFrame(data)
x = df[["StudyHours"]]
y_class = df["Pass"]
y_reg = df["Marks"]
clf = RandomForestClassifier(n_estimators=20)
clf.fit(x, y_class)
class_result = clf.predict([[5]])
print("Pass Prediction:", class_result[0])
reg = RandomForestRegressor(n_estimators=20)
reg.fit(x, y_reg)
mark_result = reg.predict([[5]])
print("Predicted Marks:", mark_result[0])
plt.scatter(df["StudyHours"], df["Marks"])
plt.plot(df["StudyHours"], reg.predict(x))
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Random Forest Regression")
plt.show()
