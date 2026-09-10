import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_samples = 1000

amount = np.random.randint(1000, 100000, n_samples)

distance_from_home = np.random.randint(1, 500, n_samples)

transaction_frequency = np.random.randint(1, 50, n_samples)

hour = np.random.randint(0, 24, n_samples)

merchant_category = np.random.choice(
            ["grocery", "electronics", "restaurant", "travel", "clothing"],
                n_samples
                )

plt.scatter(amount, distance_from_home)
plt.xlabel("Amount")
plt.ylabel("Distance from home")
plt.show()

df = pd.DataFrame({
        "amount": amount,
            "distance_from_home": distance_from_home,
                "transaction_frequency": transaction_frequency,
                    "hour": hour,
                        "merchant_category": merchant_category
                        })

print(df.info())
df.describe()


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df["merchant_category"] = encoder.fit_transform(df["merchant_category"])
X = df[
            [
                        "amount",
                                "distance_from_home",
                                        "transaction_frequency",
                                                "hour",
                                                        "merchant_category"
                                                            ]
            ]

from sklearn.ensemble import IsolationForest
model = IsolationForest(
            n_estimators=100,
                contamination=0.05,
                    random_state=42
                    )
model.fit(X)
predictions = model.predict(X)

anomalies = X[predictions == -1]

print(anomalies)
print(len(anomalies))
corr = df.corr()

print(corr)

plt.figure(figsize=(8, 6))

plt.imshow(corr, cmap="coolwarm")
plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.title("Correlation Matrix")
plt.show()
