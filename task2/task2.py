import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("/spam.csv")
print(df.head())
print(df.columns)
print(df["label"].value_counts())
df["label"].value_counts().plot(kind="bar")

plt.title("Spam vs Ham Distribution")
plt.xlabel("Label")
plt.ylabel("Number of Emails")

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
print(X.shape)
print(vectorizer.get_feature_names_out()[:20])
X = vectorizer.fit_transform(df["text"])
y = df["label"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print(X_train.shape)
print(X_test.shape)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred[:10])
print("Actual:   ", y_test.iloc[:10].values)
print("Predicted:", y_pred[:10])
