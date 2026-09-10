import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons

X, y = make_moons(
            n_samples=300,
                noise=0.08,
                    random_state=42
                    )

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.show()

from sklearn.cluster import DBSCAN
model = DBSCAN(eps=0.15, min_samples=10)
model.fit(X)

labels = model.labels_
print(labels)


plt.figure(figsize=(10, 7))

plt.scatter(
            X[:, 0],
                X[:, 1],
                    c=labels,
                        cmap=plt.cm.coolwarm
                        )

plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()

print(np.unique(labels, return_counts=True))

