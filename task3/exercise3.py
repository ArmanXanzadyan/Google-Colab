import pandas as pd
import numpy as nm

Age = nm.random.randint(11, 90, size=1000)
Annual_Income = nm.random.randint(10000, 100000, size=1000)
Spending_Score = nm.random.randint(1, 100, size=1000)
Visits_per_month = nm.random.randint(1, 30, size=1000)

df = pd.DataFrame({
    'Age': Age,
    'Annual_Income': Annual_Income,
    'Spending_Score': Spending_Score,
    'Visits_per_month': Visits_per_month
})
#print(df.info())
#print(df.isnull().sum())
#print(df.describe())
#print('Average std is\n', df.std() / df.mean())
import seaborn as sns
import matplotlib.pyplot as plt
corr_matrix = df.corr()
#plt.figure(figsize=(8, 6)) 
#sns.heatmap(
#    corr_matrix, 
#    annot=True,
#    cmap='coolwarm',                       
#    vmin=-1, vmax=1,                      
#    linewidths=0.5                          
#)

#plt.title('Correlation Matrix Heatmap')
#plt.show()
#print(df.corr())
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
#print(df_scaled)
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(df_scaled)
labels = model.labels_
print(labels)
print(pd.Series(model.labels_).value_counts().sort_index())
new_customer = [[25, 60000, 80, 15]]

new_customer_scaled = scaler.transform(new_customer)

prediction = model.predict(new_customer_scaled)

print(prediction)


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))

for cluster in range(3):
    cluster_data = df[model.labels_ == cluster]
    plt.scatter(
        cluster_data["Annual_Income"],
        cluster_data["Spending_Score"],
        label=f"Cluster {cluster}"
    )

# New customer
plt.scatter(
    new_customer[0][1],
    new_customer[0][2],
    marker="X",
    s=200,
    label="New Customer"
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("K-Means Customer Clusters")
plt.legend()
plt.show()


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertias = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(df_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(range(1, 11), inertias, marker="o")

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.xticks(range(1, 11))

plt.show()