import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
df = fetch_california_housing()
dataset = pd.DataFrame(df.data)
dataset.columns = df.feature_names
print(dataset.head())
X = dataset
y = df.target
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
comparison = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred})
print(comparison.head())
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print("r2 Score", score)
