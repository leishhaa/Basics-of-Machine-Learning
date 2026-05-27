import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8],
    "Test_Score": [50, 55, 60, 65, 70, 75, 80, 85],
    "Salary": [20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000]
}

df = pd.DataFrame(data)

x = df[["Experience", "Test_Score"]]
y = df["Salary"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = Lasso(alpha=1.0)

model.fit(x_train, y_train)

prediction = model.predict(x_test)

print("Predicted Salary =", prediction)

print("Mean Squared Error =", mean_squared_error(y_test, prediction))
print("R2 Score =", r2_score(y_test, prediction))
