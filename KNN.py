import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
data = {
    "Hours": [1, 2, 3, 4, 5, 6],
    "Marks": [20, 30, 40, 55, 70, 85],
    "Result": ["Fail", "Fail", "Fail", "Pass", "Pass", "Pass"]}
df = pd.DataFrame(data)
x = df[["Hours"]]
y_class = df["Result"]
y_reg = df["Marks"]
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x, y_class)
class_prediction = classifier.predict([[4.5]])
print("Classification Result:", class_prediction[0])
regressor = KNeighborsRegressor(n_neighbors=2)
regressor.fit(x, y_reg)
reg_prediction = regressor.predict([[4.5]])
print("Predicted Marks:", reg_prediction[0])
