import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. Create data
# ==========================================

X, _ = make_blobs(
    n_samples=1000,
    centers=4,
    n_features=2,
    cluster_std=1.2,
    random_state=42
)

np.random.seed(42)

age = np.random.randint(18, 70, 1000)

income = np.random.randint(20000, 120000, 1000)

spending_score = np.random.randint(1, 101, 1000)

visits = np.random.randint(1, 21, 1000)


# ==========================================
# 2. Create DataFrame
# ==========================================

df = pd.DataFrame({
    "Age": age,
    "Annual Income": income,
    "Spending Score": spending_score,
    "Visits per month": visits
})


print("First 5 customers:")
print(df.head())


# ==========================================
# 3. Basic information
# ==========================================

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())


# ==========================================
# 4. Missing values
# ==========================================

print("\nMissing values:")
print(df.isnull().sum())


# ==========================================
# 5. Duplicate values
# ==========================================

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ==========================================
# 6. Histograms
# ==========================================

df.hist(
    figsize=(12, 8),
    bins=20
)

plt.suptitle("Customer Features Distribution")

plt.tight_layout()

plt.show()


# ==========================================
# 7. Income vs Spending Score
# ==========================================

plt.figure(figsize=(9, 6))

plt.scatter(
    df["Annual Income"],
    df["Spending Score"],
    alpha=0.6
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.title("Annual Income vs Spending Score")

plt.show()


# ==========================================
# 8. Age vs Spending Score
# ==========================================

plt.figure(figsize=(9, 6))

plt.scatter(
    df["Age"],
    df["Spending Score"],
    alpha=0.6
)

plt.xlabel("Age")
plt.ylabel("Spending Score")

plt.title("Age vs Spending Score")

plt.show()


# ==========================================
# 9. Visits vs Spending Score
# ==========================================

plt.figure(figsize=(9, 6))

plt.scatter(
    df["Visits per month"],
    df["Spending Score"],
    alpha=0.6
)

plt.xlabel("Visits per month")
plt.ylabel("Spending Score")

plt.title("Visits vs Spending Score")

plt.show()


# ==========================================
# 10. Correlation Matrix
# ==========================================

corr = df.corr()

print("\nCorrelation Matrix:")
print(corr)


plt.figure(figsize=(8, 6))

plt.imshow(
    corr,
    cmap="coolwarm"
)

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

plt.colorbar()

plt.title("Feature Correlation")

plt.tight_layout()

plt.show()


# ==========================================
# 11. Select features for Machine Learning
# ==========================================

features = [
    "Age",
    "Annual Income",
    "Spending Score",
    "Visits per month"
]

X = df[features]


print("\nFeatures before scaling:")
print(X.head())


# ==========================================
# 12. Feature Scaling
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==========================================
# 13. Convert scaled data to DataFrame
# ==========================================

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=features
)


print("\nFeatures after scaling:")
print(X_scaled_df.head())


# ==========================================
# 14. Statistics after scaling
# ==========================================

print("\nStatistics after scaling:")
print(X_scaled_df.describe())


# ==========================================
# 15. Check mean
# ==========================================

print("\nMean after scaling:")
print(X_scaled_df.mean())


# ==========================================
# 16. Check standard deviation
# ==========================================

print("\nStandard deviation after scaling:")
print(X_scaled_df.std())


# ==========================================
# 17. Visualization BEFORE scaling
# ==========================================

plt.figure(figsize=(10, 6))

plt.boxplot(
    X.values,
    labels=features
)

plt.title("Features Before Scaling")

plt.ylabel("Original Values")

plt.xticks(rotation=20)

plt.tight_layout()

plt.show()


# ==========================================
# 18. Visualization AFTER scaling
# ==========================================

plt.figure(figsize=(10, 6))

plt.boxplot(
    X_scaled,
    labels=features
)

plt.title("Features After Standard Scaling")

plt.ylabel("Standardized Values")

plt.xticks(rotation=20)

plt.tight_layout()

plt.show()


# ==========================================
# 19. Final X for K-Means
# ==========================================

X = X_scaled

print("\nFinal X shape:")
print(X.shape)
