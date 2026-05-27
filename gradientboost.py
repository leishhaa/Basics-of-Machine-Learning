import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [20, 30, 40, 50, 65, 75, 85, 95],
    "Result": [0, 0, 0, 0, 1, 1, 1, 1]}
df = pd.DataFrame(data)
x = df[["Hours"]]
y_class = df["Result"]
y_reg = df["Marks"]
classifier = GradientBoostingClassifier()
classifier.fit(x, y_class)
class_prediction = classifier.predict([[5]])
print("Classification Result:", class_prediction[0])
regressor = GradientBoostingRegressor()
regressor.fit(x, y_reg)
reg_prediction = regressor.predict([[5]])
print("Predicted Marks:", reg_prediction[0])
