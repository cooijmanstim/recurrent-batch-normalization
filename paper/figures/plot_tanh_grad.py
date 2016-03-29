import numpy as np, matplotlib.pyplot as plt

def tanh(x): return np.tanh(x)
def dtanh(x): return 1 - tanh(x)**2
def sigmoid(x): return 1. / (1. + np.exp(-x))
def dsigmoid(x): return sigmoid(x) * (1 - sigmoid(x))

sample_size = 1000
sigmas = np.linspace(0, 1, 1000)
x = np.random.randn(sample_size, len(sigmas)) * sigmas

for fn_name, fn, dfn in [("tanh",              tanh,    dtanh),
                         ("logistic function", sigmoid, dsigmoid)]:
    y = fn(x)
    dydx = dfn(x)
    
    plt.figure()
    plt.plot(sigmas, sigmas)
    plt.plot(sigmas, y.std(axis=0))
    plt.gca().set_aspect("equal")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("input standard deviation")
    plt.ylabel("output standard deviation")
    plt.title("%s variance propagation" % fn_name)
    
    plt.figure()
    plt.plot(sigmas, dydx.mean(axis=0), color='#CC4F1B')
    plt.fill_between(sigmas,
                     np.percentile(dydx, 25, axis=0),
                     np.percentile(dydx, 75, axis=0),
                     facecolor='#FF9848', linewidth=0)
    plt.gca().set_aspect("equal")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("input standard deviation")
    plt.ylabel("expected derivative (and IQR range)")
    plt.title("derivative through %s of centered input" % fn_name)

plt.show()

