import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
data = {
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8],
    'Test_Score': [50, 55, 60, 65, 70, 75, 80, 85],
    'Salary': [20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000]
}
df = pd.DataFrame(data)
X = df[['Experience', 'Test_Score']]
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred = ridge_model.predict(X_test)

print("Predicted Values:", y_pred)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 score:", r2_score(y_test, y_pred))
