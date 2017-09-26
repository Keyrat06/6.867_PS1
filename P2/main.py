from loadFittingDataP2 import getData
import numpy as np

X, Y = getData(False)

def expand(x, m):
    x = np.array(x)
    expanded = np.zeros([len(x), m+1])
    for i in range(m+1):
        expanded[:,i] = x**i
    return expanded


def SSE_poly(y, x, theta):
    X = expand(x, len(theta)-1)
    return SSE_poly_expanded(y, X, theta)

def SSE_poly_expanded(y, X, theta):
    error = y.reshape(1, -1) - np.matmul(X, theta)
    return (error**2).sum()

def d_SSE_poly(y, x, theta):
    X = expand(x, len(theta)-1)
    return d_SSE_poly_expanded(y, X, theta)

def d_SSE_poly_expanded(y, X, theta):
    error = (y - np.matmul(X, theta))
    return error.reshape(-1, 1) * X

def gradientApprox(f, x, d):
    return (f(x+d)-f(x))/d



#with these params SSE_poly should return 0
y = np.array([3, 5, 7, 9, 11])
x = np.array([1, 2, 3, 4, 5])
theta = np.array([1, 2])
print (SSE_poly(y, x, theta))

y = np.array([4, 6, 8, 10, 12])
x = np.array([1, 2, 3, 4, 5])
theta = np.array([1, 2])
print(d_SSE_poly(y, x, theta))

f = lambda x: SSE_poly(y, x, theta)
print(gradientApprox(f, x, 0.00001))









