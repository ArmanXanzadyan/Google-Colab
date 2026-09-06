
nWith4Model(model1, model2, model3, model4):

        y_pred_1 = model1.predict(X_test)
            y_pred_2 = model2.predict(X_test)
                y_pred_3 = model3.predict(X_test)
                    y_pred_4 = model4.predict(X_test)

                        predictions = np.array([
                                    y_pred_1,
                                            y_pred_2,
                                                    y_pred_3,
                                                            y_pred_4
                                                                ])

                            final_prediction = []

                                for i in range(predictions.shape[1]):

                                            votes = predictions[:, i]

                                                    count_0 = np.sum(votes == 0)
                                                            count_1 = np.sum(votes == 1)

                                                                    if count_1 > count_0:
                                                                                    final_prediction.append(1)
                                                                                            else:
                                                                                                            final_prediction.append(0)

                                                                                                                return np.array(final_prediction)

















                                                                                                            from sklearn.metrics import accuracy_score
                                                                                                            from sklearn.metrics import precision_score
                                                                                                            from sklearn.metrics import recall_score
                                                                                                            from sklearn.metrics import f1_score
                                                                                                            from sklearn.metrics import confusion_matrix
                                                                                                            def foo(y_test, y_pred):
                                                                                                                  print('accuracy_score', accuracy_score(y_test, y_pred)), 
                                                                                                                    print('precision_score', precision_score(y_test, y_pred))
                                                                                                                      print('recall_score', recall_score(y_test, y_pred))
                                                                                                                        print('f1_score', f1_score(y_test, y_pred))
                                                                                                                          print('confusion_matrix\n', '\n', confusion_matrix(y_test, y_pred))




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

                                                                                                                          X = df.drop('target', axis=1)
                                                                                                                          y = df['target']


                                                                                                                          from sklearn.model_selection import train_test_split

                                                                                                                          X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                                                                                                                          from sklearn.ensemble import RandomForestClassifier

                                                                                                                          model1 = RandomForestClassifier(n_estimators=101, random_state=42)

                                                                                                                          model1.fit(X_train, y_train)

                                                                                                                          y_pred = model1.predict(X_test)
                                                                                                                          print('RANDOM_FOREST_INFO', '\n')
                                                                                                                          foo(y_test, y_pred)
                                                                                                                          print('\n', 40 * '*')

                                                                                                                          from sklearn.tree import DecisionTreeClassifier

                                                                                                                          model2 = DecisionTreeClassifier(random_state=42)

                                                                                                                          model2.fit(X_train, y_train)

                                                                                                                          y_pred = model2.predict(X_test)
                                                                                                                          print('DECISION_TREE_INFO', '\n')
                                                                                                                          foo(y_test, y_pred)

                                                                                                                          print('\n', 40 * '*')

                                                                                                                          from sklearn.linear_model import LogisticRegression

                                                                                                                          model3 = LogisticRegression(random_state=42)

                                                                                                                          model3.fit(X_train, y_train)

                                                                                                                          y_pred = model3.predict(X_test)

                                                                                                                          print('LOGISTIC_REGRESSION_INFO', '\n')
                                                                                                                          foo(y_test, y_pred)

                                                                                                                          print('\n', 40 * '*')

                                                                                                                          from sklearn.neighbors import KNeighborsClassifier

                                                                                                                          model4 = KNeighborsClassifier()

                                                                                                                          model4.fit(X_train, y_train)

                                                                                                                          y_pred = model4.predict(X_test)

                                                                                                                          print('KNN_INFO', '\n')
                                                                                                                          foo(y_test, y_pred)

                                                                                                                          print('\n', 40 * '*')

                                                                                                                          majority_vote = predictionWith4Model(model1, model2, model3, model4)

                                                                                                                          print('MAJORITY_VOTE_INFO', '\n')
                                                                                                                          foo(y_test, majority_vote)
