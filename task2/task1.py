import pandas as pd

data = {
    "area": [50, 60, 70, 80, 90, 100, 120, 140, 160, 180],
    "bedrooms": [1, 2, 2, 2, 3, 3, 3, 4, 4, 5],
    "age": [20, 15, 10, 8, 12, 5, 7, 4, 3, 2],
    "location_score": [5, 6, 6, 7, 7, 8, 8, 9, 9, 10],
    "price": [100, 125, 145, 165, 190, 220, 250, 300, 340, 400]
}

df = pd.DataFrame(data)

print(df)
print(df.describe())
print(df.isnull().sum())
import matplotlib.pyplot as plt

plt.scatter(df["area"], df["price"])

plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs Price")

plt.show()
print(df.corr())


import seaborn as sns

plt.figure(figsize=(8, 6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()
from sklearn.model_selection import train_test_split

X = df[["area", "bedrooms", "age", "location_score"]]

y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Actual:", y_test.values)
print("Predicted:", y_pred)
