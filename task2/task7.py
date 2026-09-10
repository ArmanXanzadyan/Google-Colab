import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

df = pd.DataFrame({
    "Goals": np.random.randint(0, 30, 100),
    "Assists": np.random.randint(0, 20, 100),
    "Pass Accuracy": np.random.randint(60, 96, 100),
    "Tackles": np.random.randint(0, 10, 100),
    "Dribbles": np.random.randint(0, 15, 100),
    "Shots": np.random.randint(10, 100, 100)
})


print(df.head())
print(df.info())
print(df.describe())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df)

print(X_scaled[:5])

from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)
df["Cluster"] = clusters

print(df.head())

cluster_stats = df.groupby("Cluster").mean()

print(cluster_stats)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print(X_pca[:5])

plt.figure(figsize=(10, 7))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Football Players Clusters")

plt.show()
