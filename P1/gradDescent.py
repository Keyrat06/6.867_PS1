import loadParametersP1 as lp
import numpy as np



#Implementing basic gradient descent
def gradDescent(n, dg, step_size, threshold, num_iterations, theta = None):
    if theta is None:
        theta = np.zeros(n)

    for i in range(num_iterations):

        new_theta = theta - step_size * dg(theta)

        if np.linalg.norm(new_theta - theta) < threshold:
            return theta

        theta = new_theta

    return theta

def d_quadraticBowl(x):
    A = 0
    b = 0
    pass


mu, S, A, b = lp.getData()

def d_gaussian(x):
    n = len(mu)
    return -1 * gaussian(x, mu, S, n) * np.linalg.inv(S) * (x-mu)

def gaussian(x, mu, S, n):
    coeff = -(10**4)/np.sqrt((2*np.pi)**n * np.linalg.norm(S))
    exponent = -0.5 * (x-mu).T * np.linalg.inv(S) * (x-mu)
    return coeff * np.exp(exponent)

