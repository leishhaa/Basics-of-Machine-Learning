import pandas as pd
from sklearn.naive_bayes import GaussianNB

data = {
    "Hours": [1, 2, 3, 4, 5, 6],
    "Result": ["Fail", "Fail", "Fail", "Pass", "Pass", "Pass"]
}

df = pd.DataFrame(data)

x = df[["Hours"]]
y = df["Result"]

model = GaussianNB()

model.fit(x, y)

prediction = model.predict([[4.5]])

print("Predicted Result:", prediction[0])
    
