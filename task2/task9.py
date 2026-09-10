import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.DataFrame({
        'age': np.random.randint(18, 90, 1000),

            'blood_pressure': np.random.randint(90, 181, 1000),

                'cholesterol': np.random.randint(120, 301, 1000),

                    'BMI': np.random.uniform(15, 40, 1000),

                        'exercise_frequency': np.random.randint(0, 8, 1000)
                        })

print(df.info())

print(df.describe())

df.hist(bins=50, figsize=(20,15))
plt.show()
df['target'] = np.where(
            (
                        (df['age'] > 60) &
                                (df['blood_pressure'] > 140) &
                                        (df['BMI'] > 30)
                                            ) |
                (
                            (df['cholesterol'] > 240) &
                                    (df['exercise_frequency'] < 3)
                                        ),
                    1,
                        0
                        )
print(df.head())
print(df['target'].value_counts())

X = df.drop('target', axis=1)
y = df['target']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=101, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
from sklearn.metrics import precision_score
print(precision_score(y_test, y_pred))
from sklearn.metrics import recall_score
print(recall_score(y_test, y_pred))
from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

print(df.corr())

import seaborn as sns

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')


feature_importance = pd.DataFrame({
        'feature': X.columns,
            'importance': model.feature_importances_
            })

print(feature_importance)
