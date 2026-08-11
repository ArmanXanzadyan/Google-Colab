from sklearn.linear_model import LinearRegression

# Մեր տվյալները
X = [[1], [2], [3.6], [4], [5]]
y = [2, 4, 7.2, 8, 10]

# Ստեղծում ենք model
model = LinearRegression()

# Սովորեցնում ենք model-ին
model.fit(X, y)

# Նոր տվյալ
X_new = [[6]]

# Prediction
prediction = model.predict(X_new)

print(prediction)

