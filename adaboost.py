import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Result": [0, 0, 0, 0, 1, 1, 1, 1]
}
df = pd.DataFrame(data)
x = df[["Hours"]]
y = df["Result"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1
)
model = AdaBoostClassifier(n_estimators=10)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
prediction = model.predict([[5]])
print("Prediction:", prediction[0])
