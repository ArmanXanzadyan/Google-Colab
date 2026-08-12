import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
years = [1, 2, 3, 4, 5, 6]
salary = [300, 350, 400, 450, 500, 550]

df = pd.DataFrame({
            'years': years,
                        'Salary': salary
                                    })
print(df)
plt.scatter(years, salary)
plt.xlabel('Years')
plt.ylabel('Salary')
plt.title('Experience')
plt.show()
X = df[['years']]
y = df['Salary']

model = LinearRegression()
model.fit(X, y)
prediction = model.predict([[7]])
print(prediction)
plt.scatter(years, salary)
plt.xlabel('Years')
plt.ylabel('Salary')
plt.title('Experience')
plt.show()
