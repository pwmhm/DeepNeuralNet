import numpy as np

class loss():
    def __init__(self):
        self.loss_mean = []
        self.loss_modules = []
        self.weights=[]
    def __call__(self, pred, truth):
        _loss=0
        for modules, weight in zip(self.loss_modules, self.weights):
            _loss += modules(pred,truth) * weight
        self.loss_mean.append(_loss)
        return _loss

    def clear(self):
        self.loss_mean.clear()
    
    def get_mean(self):
        return np.mean(self.loss_mean)

    def _backward(self, pred, truth):
        pass


    def add(self, loss, weight):
        self.loss_modules.append(loss)
        self.weights.append(weight)

def mse(pred, truth):
    retval = np.power((pred-truth), 2)
    return np.mean(retval)

def mse_derivative(pred, truth):
    retval = 2*(pred-truth)
    return np.mean(retval)


def rmse(pred, truth):
    _mse = mse(pred, truth)
    return np.sqrt(_mse)
