import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Result": [0, 0, 0, 0, 1, 1, 1, 1]}
df = pd.DataFrame(data)
x = df[["Hours"]]
y = df["Result"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1)
pre_model = DecisionTreeClassifier(max_depth=2)
pre_model.fit(x_train, y_train)
pre_pred = pre_model.predict(x_test)
print("Pre Pruning Accuracy:", accuracy_score(y_test, pre_pred))
post_model = DecisionTreeClassifier(ccp_alpha=0.01)
post_model.fit(x_train, y_train)
post_pred = post_model.predict(x_test)
print("Post Pruning Accuracy:", accuracy_score(y_test, post_pred))
