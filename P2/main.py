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
    X = expand(x, len(theta))
    error = y - np.matmul(X, theta)

