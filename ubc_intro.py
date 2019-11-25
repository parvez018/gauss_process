from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def kernel(a,b):
    sqdist = np.sum(a**2,1).reshape(-1,1)+np.sum(b**2,1)-2*np.dot(a,b.T)
    l = 0.8
    return np.exp(-(1.0/(2*l*l))*sqdist)

def pkernel(a,b,param):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * sqdist)

def gp_prior():
    n = 20
    total_function = 5
    xlimit = 3
    xtest = np.linspace(-xlimit,xlimit,n).reshape(-1,1)
    k_ = kernel(xtest,xtest)
    L = np.linalg.cholesky(k_+1e-6*np.eye(n))
    rpoints = np.random.normal(size=(n,total_function))
    fprior = np.dot(L,rpoints)
    plt.plot(xtest,fprior)
    plt.show()

def gp_posterior_zmean(ax=None):
    param = 0.1
    n = 100
    Xtest = np.linspace(-5, 5, n).reshape(-1,1)
    K_ss = pkernel(Xtest, Xtest, param)
    # Noiseless training data
    Xtrain = np.array([-4, -3, -2, -1, 1]).reshape(5,1)
    # ytrain = np.sin(Xtrain)
    # ytrain = np.power(Xtrain,2)
    ytrain = np.sin(Xtrain)+2
    ytrue = np.sin(Xtest)+2


    # Apply the kernel function to our training points
    K = pkernel(Xtrain, Xtrain, param)
    L = np.linalg.cholesky(K + 0.00005*np.eye(len(Xtrain)))

    # Compute the mean at our test points.
    K_s = pkernel(Xtrain, Xtest, param)
    Lk = np.linalg.solve(L, K_s)
    mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((n,))

    # Compute the standard deviation so we can plot it
    s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
    stdv = np.sqrt(s2)
    # Draw samples from the posterior at our test points.
    L = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
    f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,3)))

    ax.plot(Xtrain, ytrain, 'bs', ms=6, label='Training samples')
    ax.plot(Xtest,ytrue, 'g-', ms=2,label='True function value')
    ax.plot(Xtest, mu, 'r--', lw=2, label='Predicted function value')
    ax.legend(loc="lower right")
    # ax.plot(Xtest, f_post)
    ax.fill_between(Xtest.flat, mu-2*stdv, mu+2*stdv, color="#dddddd")
    
    # ax.axis([-5, 5, -3, 3])
    ax.set_title('Zero mean GP posterior')


def gp_posterior_nzmean(ax=None):
    param = 0.1
    n = 100
    mu = np.zeros(n)
    mu = mu + 2
    print("mu",np.shape(mu))
    print(mu)
    Xtest = np.linspace(-5, 5, n).reshape(-1,1)
    K_ss = pkernel(Xtest, Xtest, param)
    # Noiseless training data
    Xtrain = np.array([-4, -3, -2, -1, 1]).reshape(-1,1)
    # ytrain = np.power(Xtrain,2)
    ytrain = np.sin(Xtrain)+2
    ytrue = np.sin(Xtest)+2

    # Apply the kernel function to our training points
    K = pkernel(Xtrain, Xtrain, param)
    L = np.linalg.cholesky(K + 0.00005*np.eye(len(Xtrain)))

    # Compute the mean at our test points.
    K_s = pkernel(Xtrain, Xtest, param)
    Lk = np.linalg.solve(L, K_s)
    ytrain = ytrain - 2
    mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((n,)) + 2
    ytrain = ytrain + 2
    # print("mu",np.shape(mu))

    # Compute the standard deviation so we can plot it
    s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
    stdv = np.sqrt(s2)
    # Draw samples from the posterior at our test points.
    # L = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
    # f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,3)))

    # ax.plot(Xtrain, ytrain, 'bs', ms=8)
    # ax.plot(Xtest, f_post)
    ax.fill_between(Xtest.flat, mu-2*stdv, mu+2*stdv, color="#dddddd")
    # ax.plot(Xtest, mu, 'r--', lw=2)
    # ax.axis([-5, 5, -3, 3])
    ax.plot(Xtrain, ytrain, 'bs', ms=6, label='Training samples')
    ax.plot(Xtest,ytrue, 'g-', ms=2,label='True function value')
    ax.plot(Xtest, mu, 'r--', lw=2, label='Predicted function value')
    ax.legend(loc="lower right")
    ax.set_title('Non-zero mean (mean=2) GP posterior')

def show_plots():
    total = 2
    f, axarr = plt.subplots(1,total,figsize=(12, 5),num="True function = 2+sin(x)")
    gp_posterior_zmean(axarr[0])
    gp_posterior_nzmean(axarr[1])
    # plt.set_title("afg")
    # plt.set_window_title('gt')
    plt.show()

if __name__ == "__main__":
    print("Gaussian process")
    np.random.seed(0)
    show_plots()
    