import tarfile
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import seaborn as sns
import matplotlib.pyplot as plt


with tarfile.open("housing.tgz", "r:gz") as tar:
    tar.extractall("/")


df = pd.read_csv("/housing/housing.csv")

print(df.head())
print(df.info())
print(df.describe())


df["income_cat"] = pd.cut(
    df["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

count = df["income_cat"].value_counts()

print(count)

bar_chart = count.plot.bar()
bar_chart.set_xlabel("Income Category")
bar_chart.set_ylabel("Count")

percents = count / count.sum() * 100

print(percents)


train_set, test_set = train_test_split(
    df,
    test_size=0.2,
    stratify=df["income_cat"],
    random_state=42
)


print(df["income_cat"].value_counts(normalize=True) * 100)
print(train_set["income_cat"].value_counts(normalize=True) * 100)
print(test_set["income_cat"].value_counts(normalize=True) * 100)


for set_ in (train_set, test_set):
    set_.drop("income_cat", axis=1, inplace=True)


corr_matrix = train_set.corr(numeric_only=True)

plt.figure(figsize=(12, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")
plt.show()


print(
    corr_matrix["median_house_value"]
    .sort_values(ascending=False)
)


print(test_set.isnull())
print(test_set.isnull().sum())


median = train_set["total_bedrooms"].median()

train_set["total_bedrooms"] = train_set["total_bedrooms"].fillna(median)

test_set["total_bedrooms"] = test_set["total_bedrooms"].fillna(median)


print(train_set["total_bedrooms"].isnull().sum())
print(test_set["total_bedrooms"].isnull().sum())


encoder = OneHotEncoder()

encoder.fit(train_set[["ocean_proximity"]])

encoded = encoder.transform(
    train_set[["ocean_proximity"]]
)


print(encoded.toarray())
print(encoder.categories_)
print(encoder.get_feature_names_out())


feature_names = encoder.get_feature_names_out()

train_set[feature_names] = encoded.toarray()

test_set[feature_names] = encoder.transform(
    test_set[["ocean_proximity"]]
).toarray()


print(train_set.head())


train_set.drop("ocean_proximity", axis=1, inplace=True)
test_set.drop("ocean_proximity", axis=1, inplace=True)


print(train_set.head())
print(train_set.shape)
print(test_set.shape)
X_train = train_set.drop("median_house_value", axis=1)
y_train = train_set["median_house_value"]
X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"]
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_train)

print(predictions[:10])
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_train, predictions))

print(rmse)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_train, predictions)

print(mae)

test_predictions = model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print("Test MAE:", test_mae)
print("Test RMSE:", test_rmse)

from sklearn.metrics import r2_score

train_r2 = r2_score(y_train, predictions)
test_r2 = r2_score(y_test, test_predictions)

print("Train R²:", train_r2)
print("Test R²:", test_r2)
print(model.coef_)
print(model.intercept_)
print(X_train.columns)