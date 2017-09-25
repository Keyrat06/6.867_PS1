import loadParametersP1 as lp
import numpy as np
import matplotlib.pyplot as plt

#Implementing basic gradient descent
def gradDescent(n, dg, step_size, threshold, num_iterations, theta = None):
    if theta is None:
        theta = np.zeros(n)

    for i in range(num_iterations):
        print(theta)
        new_theta = theta - step_size * dg(theta)
        print(new_theta)
        if np.linalg.norm(new_theta - theta) < threshold:
            return theta

        theta = new_theta

    return theta

def d_quadraticBowl(x):
    return (np.matmul(x, A)-b)

def quadraticBowl(x):
    return .5 * (x * np.matmul(A, x.T)) - x.T * b

def d_gaussian(x):
    n = len(mu)
    return -1 * np.matmul(np.matmul(gaussian(x, mu, S, n), np.linalg.inv(S)), (x-mu))

def gaussian(x, mu, S, n):
    coeff = -(1)/np.sqrt((2*np.pi)**n * np.linalg.norm(S))
    exponent = -0.5 * (x-mu).T * np.linalg.inv(S) * (x-mu)
    return coeff * np.exp(exponent)

def l_gaussian(mu, S, n):
    return lambda x: gaussian(x, mu, S, n)

def gradientApprox(f, x, d):
    return (f(x+d)-f(x))/d


mu, S, A, b = lp.getData()
n = len(mu)

## part 1
# print(mu, S, A, b)
# n = len(mu)
# a = gradDescent(n, d_gaussian, 100, 0.00001, 20)
# print(a)
# b = gradDescent(n, d_quadraticBowl, 0.1, 0.00001, 20)
# print(b)


## part 2
d = 0.000000000001
g_differences = []
q_differences = []
for x in np.random.random([100, n]):
    x * 100
    g_differences.append(np.linalg.norm(gradientApprox(l_gaussian(mu, S, n), x, d) - d_gaussian(x)))
    q_differences.append(np.linalg.norm(gradientApprox(quadraticBowl, x, d) - d_quadraticBowl(x)))
plt.hist(g_differences)
plt.title("g_diff")
plt.show('hold')
plt.hist(q_differences)
plt.title("q_diff")
plt.show('hold')
