import matplotlib.pyplot as plt
import pylab as pl
import numpy as np

def getData(ifPlotData=True):
    # load the fitting data and (optionally) plot out for examination
    # return the X and Y as a tuple

    data = pl.loadtxt('curvefittingp2.txt')

    X = data[0,:]
    Y = data[1,:]

    if ifPlotData:
        x_data = np.linspace(0, 1, 100)
        y_data = np.cos(np.pi*x_data) + np.cos(2*np.pi*x_data)
        plt.plot(x_data,y_data, color="orange", label = "sampled distribution")
        plt.plot(X, Y, 'o', label="training points")
        plt.xlabel('x')
        plt.ylabel('y')
        #plt.show()

    return (X,Y)
