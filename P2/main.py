from P2.loadFittingDataP2 import getData
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl


# X, Y = getData(False)

def expand(x, m):
    x = np.array(x)
    expanded = np.zeros([len(x), m+1])
    for i in range(m+1):
        expanded[:, i] = (x**i).reshape(-1)
    return expanded

def expand_theta(x, m):
    x = np.array(x)
    expanded = np.zeros([len(x), m+1])
    for i in range(m+1):
        expanded[:, i] = np.cos(i*np.pi*x)
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
    error = np.matmul(X, theta) - y.reshape(1, -1)
    return 2*(error.T * X).sum(axis=0)

def gradientApprox(f, x, d):
    return (f(x+d)-f(x))/d

def maxLikelihoodVector(x,y,m):
    x_panded = expand(x,m)
    x_panded = np.matrix(x_panded)
    y = np.matrix(y)
    w = np.linalg.inv(x_panded.T*x_panded)*x_panded.T*y.T
    return w

#Batch gradient descent for problem 2.3
def batchGradDescent(y, x, m, step_size, num_iterations, theta = None, expander = expand):
    x = expander(x,m)
    if theta is None:
        theta = np.zeros([len(x[0])])
    evaluations = []
    thetas = []
    for i in range(num_iterations):
        evaluations.append(i*len(y))
        thetas.append(theta.sum())

        d_sse_error = d_SSE_poly_expanded(y, x, theta)
        theta -= step_size * d_sse_error

    return theta, (evaluations, thetas)

#Batch gradient descent for problem 2.3
def stochGradDescent(y, x, m, t_o, k, num_iterations, theta = None, expander = expand):
    x = expander(x, m)
    if theta is None:
        theta = np.zeros(len(x[0]))
    thetas = []
    evals = 0
    evaluations = []
    for i in range(num_iterations):
        p = np.random.permutation(len(y))
        y = y[p]
        x = x[p]
        step_size = (t_o+i)**(-k)
        for point in range(len(x)):
            thetas.append(theta.sum())
            evaluations.append(evals)
            evals += 1
            theta -= step_size * d_SSE_poly_expanded(y[point], x[point], theta)
    return theta, (evaluations, thetas)

# # part 1
# X, Y = getData(True)
# X_plot = np.linspace(0, 1, 200)
#
# theta_0 = maxLikelihoodVector(X, Y, 0)
# sol_0 = np.matmul(expand(X_plot, len(theta_0)-1), theta_0)
#
# theta_1 = maxLikelihoodVector(X, Y, 1)
# sol_1 = np.matmul(expand(X_plot, len(theta_1)-1), theta_1)
#
# theta_3 = maxLikelihoodVector(X, Y, 3)
# sol_3 = np.matmul(expand(X_plot, len(theta_3)-1), theta_3)
#
# theta_10 = maxLikelihoodVector(X, Y, 10)
# sol_10 = np.matmul(expand(X_plot, len(theta_10)-1), theta_10)
#
#
# plt.plot(X_plot, sol_0, color="blue", label="m=0")
# plt.plot(X_plot, sol_1, color="green", label="m=1")
# plt.plot(X_plot, sol_3, color="red", label="m=3")
# plt.plot(X_plot, sol_10, color="purple", label="m=10")
# plt.legend()
# plt.title("Fitting data with polynomials of different degrees")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show("hold")

##
#part 3

# m1 = 2
# m2 = 5
# X, Y = getData(True)
# X_plot = np.linspace(0, 1, 200)
#
# theta_batch_poly, e_batch_poly = batchGradDescent(Y, X, m1, .05, 300)
# sol_batch_poly = np.matmul(expand(X_plot, len(theta_batch_poly)-1), theta_batch_poly)
# theta_batch_poly2, e_batch_poly2 = batchGradDescent(Y, X, m2, .05, 300)
# sol_batch_poly2 = np.matmul(expand(X_plot, len(theta_batch_poly2)-1), theta_batch_poly2)
#
# theta_stoch_poly, e_stoch_poly = stochGradDescent(Y, X, m1, 100, 0.5, 300)
# sol_stoch_poly = np.matmul(expand(X_plot, len(theta_stoch_poly)-1), theta_stoch_poly)
# theta_stoch_poly2, e_stoch_poly2 = stochGradDescent(Y, X, m2, 100, 0.5, 300)
# sol_stoch_poly2 = np.matmul(expand(X_plot, len(theta_stoch_poly2)-1), theta_stoch_poly2)
#
# plt.plot(X_plot, sol_batch_poly, color="yellow", label="Batch polynomial m = 2")
# plt.plot(X_plot, sol_stoch_poly, color="blue", label="Stochastic polynomial m = 2")
# plt.plot(X_plot, sol_batch_poly2, color="red", label="Batch polynomial m = 5")
# plt.plot(X_plot, sol_stoch_poly2, color="purple", label="Stochastic polynomial m = 5")
# print((sol_batch_poly2-sol_stoch_poly2).sum())
# plt.title("batch vs stochastic gradient decent")
# plt.xlabel("number of evaluations")
# plt.ylabel("sum of theta vector")
# plt.legend()
# plt.show("hold")


## part 4
m = 8
X, Y = getData(False)
X_plot = np.linspace(0, 1, 200)

theta_batch, e_batch = batchGradDescent(Y, X, m, .05, 300, expander=expand_theta)
theta_stoch, e_stoch = stochGradDescent(Y, X, m, 100, 0.5, 300, expander=expand_theta)

sol_batch = np.matmul(expand(X_plot, m), theta_batch)
sol_stoch = np.matmul(expand(X_plot, m), theta_stoch)



# plt.plot(e_batch[0], e_batch[1], label="Batch polynomial")
# plt.plot(e_stoch[0], e_stoch[1], label="Stochastic polynomial")

# plt.plot(X_plot, sol_batch, color="red", label="Batch cosine")
# plt.plot(X_plot, sol_stoch, color="green", label="Stochastic polynomial")

print(theta_batch)
print(theta_stoch)
real_values = [0,1,1,0,0,0,0,0,0]


# Four polar axes
plt.subplot(1, 3, 1)
plt.title("Batch weights")
plt.bar(np.arange(len(theta_batch)), theta_batch)
plt.xlabel("Theta Term")
plt.ylabel("weight")

plt.subplot(1, 3, 2)
plt.title("Stoch weights")
plt.bar(np.arange(len(theta_stoch)), theta_stoch)
plt.xlabel("Theta Term")
plt.ylabel("weight")
plt.subplot(1, 3, 3)

plt.title("Sampled weights")
plt.bar(np.arange(len(real_values)), real_values)
plt.xlabel("Theta Term")
plt.ylabel("weight")


plt.subplots_adjust(hspace=0.3)

plt.show("hold")


# #
# theta_batch_poly, e_batch_poly = batchGradDescent(Y, X, m, .05, 300)
# sol_batch_poly = np.matmul(expand(X_plot, len(theta_batch_poly)-1), theta_batch_poly)
#
# theta_stoch_poly, e_stoch_poly = stochGradDescent(Y, X, m, 100, 0.5, 300)
# sol_stoch_poly = np.matmul(expand(X_plot, len(theta_stoch_poly)-1), theta_stoch_poly)
#
# theta_batch_cos, e_batch_cos = batchGradDescent(Y, X, m, .001, 300, expander=expand_theta)
# sol_batch_cos = np.matmul(expand_theta(X_plot, len(theta_batch_cos)-1), theta_batch_cos)
#
# theta_stoch_cos, e_stoch_cos = stochGradDescent(Y, X, m, 20, 0.75, 300, expander=expand_theta)
# sol_stoch_cos = np.matmul(expand_theta(X_plot, len(theta_stoch_cos)-1), theta_stoch_cos)
# # plot values
#
# plt.plot(e_batch_poly[0], e_batch_poly[1], label="Batch polynomial")
# plt.plot(e_stoch_poly[0], e_stoch_poly[1], label="Stochastic polynomial")
# plt.plot(e_batch_cos[0], e_batch_cos[1], label="Batch Cosine")
# plt.plot(e_stoch_cos[0], e_stoch_cos[1], label="Stochastic Cosine")
# plt.title("batch vs stochastic gradient decent")
# plt.xlabel("number of evaluations")
# plt.ylabel("sum of theta vector")
# plt.legend()
# plt.show("hold")
# #
# plot fittings #remember to turn plot to true in getData
# plt.plot(X_plot, sol_batch_poly, color="blue", label="Batch polynomial")
# plt.plot(X_plot, sol_stoch_poly, color="green", label="Stochastic polynomial")
# plt.plot(X_plot, sol_batch_cos, color="red", label="Batch cosine")
# plt.plot(X_plot, sol_stoch_cos, color="purple", label="Stochastic cosine")
# plt.legend()
# plt.title("Fitting data with polynomial or cosines M = 3")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show("hold")

