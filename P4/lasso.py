from P4 import lassoData
from sklearn.linear_model import Lasso
from P2.main import expand

x_train, y_train = lassoData.lassoTrainData()
x_valid, y_valid = lassoData.lassoValData()
x_test, y_test = lassoData.lassoTestData()

print(x_train, y_train)

X_train = expand(x_train, 5)
X_valid = expand(x_valid, 5)
X_test = expand(x_test, 5)

train_x, train_y = lassoData.lassoTrainData()

alpha = 0.1
lasso = Lasso(alpha=alpha)  # alpha is lambda
y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)

