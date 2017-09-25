import loadParametersP1 as lp
import numpy as np

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


mu, S, A, b = lp.getData()

def d_quadraticBowl(x):
    return (np.matmul(x, A)-b)
def quadraticBowl(x):
    return .5*np.matmul(x,np.matmul(A,x.T))-np.matmul(x.T,b)

def d_gaussian(x):
    n = len(mu)
    return -1 * np.matmul(np.matmul(gaussian(x, mu, S, n),np.linalg.inv(S)), (x-mu))

def gaussian(x, mu, S, n):
    coeff = -(1)/np.sqrt((2*np.pi)**n * np.linalg.norm(S))
    exponent = -0.5 * (x-mu).T * np.linalg.inv(S) * (x-mu)
    return coeff * np.exp(exponent)

if __name__ == "__main__":
    print(mu, S, A, b)
    n = len(mu)
    #a = gradDescent(n, d_gaussian, 100, 0.00001, 20)
    #print(a)
    b = gradDescent(n, d_quadraticBowl, 0.1, 0.00001, 20)
    print(b)
