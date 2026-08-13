import numpy as np
import matplotlib.pyplot as plt
area = np.array([50, 60, 70, 80, 90])
price = np.array([100, 120, 140, 160, 180])
print('The average is area is ', np.mean(area))
print('The min area is ', np.min(area))
print('The max area is ', np.max(area))
print('The min price is ', np.min(price))
print('The max price is ', np.max(price))
plt.scatter(area, price)
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('House Prices')
plt.show()

