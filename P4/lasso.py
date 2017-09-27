from P4 import lassoData
from sklearn.linear_model import Lasso
from P2.main import expand, plot
import matplotlib.pyplot as plt
import numpy as np

x_train, y_train = lassoData.lassoTrainData()
x_valid, y_valid = lassoData.lassoValData()
x_test, y_test = lassoData.lassoTestData()

print(x_train, y_train)

X_train = expand(x_train, 10)
X_valid = expand(x_valid, 10)
X_test = expand(x_test, 10)

train_x, train_y = lassoData.lassoTrainData()

alpha = 0.1
lasso = Lasso(alpha=alpha)  # alpha is lambda
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

theta = lasso.coef_
X_test = np.linspace(-1, 1, 100)
sol = np.matmul(expand(X_test, len(theta)-1), theta)

plt.plot(X_test, sol)
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x_train, y_train, 'o')
plt.show()
