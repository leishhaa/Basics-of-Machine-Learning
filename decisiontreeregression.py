import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8],
    "Salary": [25000, 30000, 40000, 50000, 60000, 70000, 80000, 90000]}
df = pd.DataFrame(data)
x = df[["Experience"]]
y = df["Salary"]
model = DecisionTreeRegressor()
model.fit(x, y)
prediction = model.predict([[9]])
print("Predicted Salary:", prediction[0])
plt.scatter(x, y)
plt.plot(x, model.predict(x))
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Decision Tree Regression")
plt.show()
