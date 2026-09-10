import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(42)

df = pd.DataFrame({
        "income": np.random.randint(20000, 100000, 200),
            "age": np.random.randint(20, 60, 200),
                "credit_score": np.random.randint(500, 850, 200),
                    "loan_amount": np.random.randint(5000, 100000, 200),
                        "employment_years": np.random.randint(0, 20, 200)
                        })

plt.scatter(df["credit_score"], df["income"])
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("Age vs Income")
plt.show()


sns.pairplot(df)
plt.show()

df["target"] = np.where(
            (df["credit_score"] > 700) & (df["income"] > 50000),
                "Approved",
                    "Rejected"
                    )
print(df.head())

X = df.drop("target", axis=1)
y = df["target"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
            X, y,
                test_size=0.2,
                    random_state=42
                    )

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
            random_state=42
            )
model.fit(X_train, y_train)


predictions = model.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print(accuracy)



from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))

plot_tree(
            model,
                feature_names=X.columns,
                    class_names=model.classes_,
                        filled=True
                        )

plt.show()
