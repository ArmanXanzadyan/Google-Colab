
import sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
Y = iris.target

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


DF = pd.DataFrame(X, columns=iris.feature_names)
DF["target"] = Y
print(DF)

plt.scatter(DF["sepal length (cm)"], Y)
sns.pairplot(DF, diag_kind="hist")
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
            X,
                Y,
                    test_size=0.2,
                        random_state=42
                        )
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, Y_train)

predictions = model.predict(X_test)

print(predictions)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test, predictions)

print(accuracy)
