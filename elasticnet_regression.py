import pandas as pd
from sklearn.linear_model import ElasticNet

data = {
    "Experience": [1, 2, 3, 4, 5],
    "Salary": [25000, 30000, 40000, 50000, 60000]
}

df = pd.DataFrame(data)

x = df[["Experience"]]
y = df["Salary"]

model = ElasticNet()

model.fit(x, y)

prediction = model.predict([[6]])

print("Predicted Salary:", prediction[0])
