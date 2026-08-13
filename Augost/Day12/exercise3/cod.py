import pandas as pd
import numpy as np


data = {
            'Name': ['Anna', 'Bob', 'John', 'Mary', 'David'],
                'Age': [20, 21, 19, 22, 20],
                    'Math': [85, 70, 95, 60, 88],
                        'Python': [90, 80, 92, 75, 85]
                        }
df = pd.DataFrame(data)
print(df)
print(40 * '*')
a = df['Math']
print(a)
print(40 * '*')
print('Average result Math is', np.mean(a))
print(40 * '*')
print('Max grades is Python is ', np.max(df['Python']))
print(df[df['Math'] > 80][['Name', 'Math']])
df['Average'] = (df['Math'] + df['Python']) / 2
print(df)
