import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 85],
    "Pass": [0, 0, 0, 0, 1, 1, 1, 1]
}
df = pd.DataFrame(data)
x = df[["Hours_Studied", "Attendance"]]
y = df["Pass"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=4)

model = LogisticRegression()
model.fit(x_train, y_train)
prediction = model.predict(x_test)

print("Predicted Output =", prediction)

print("Actual Output =", y_test.values)

print("Accuracy =", accuracy_score(y_test, prediction))
