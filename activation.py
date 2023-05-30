import numpy as np


def binary(z, thresh=0.5):
  return np.where(z<thresh, 0, z)

def relu(z):
  return np.where(z<0, 0, z)

def leaky_relu(z, alpha=0.1):
  return np.where(z<0, alpha*z, z)

def elu(z, alpha=0.1):
  return np.where(z<0, alpha*(np.exp(z)-1), z)

def softmax(z):
  _exp = np.exp(z)
  _denom = np.sum(_exp, axis=0)

  return [np.divide(exp, _denom) for exp in _exp]

def sigmoid(z):
  return [np.divide(1, (1+np.exp(-z)))]

def binary_derivative(z):
  return [0]

def relu_derivative(z):
  return np.where(z<0, 0, 1)

def leaky_relu_derivative(z, alpha=0.1):
  return np.where(z<0, alpha, 1)

def elu_derivative(z, alpha=0.1):
  return np.where(z<0, alpha, 1)

def softmax_derivative(z):
  # create jacobian matrix of nxn dimension
  # jnm = softmax(n) * ( delta_nm - softmax(m)) ; delta_nm = 1 if i=j else 0

  _exp = np.exp(z)
  _denom = np.sum(_exp, axis=0)

  _jacobian = np.zeros((len(z), len(z)))
  for n in range(len(z)):
    for m in range(len(z)):
        if n == m:
          _jacobian[n, m] = (_exp[n]/_denom)*(1-(_exp[m]/_denom))
        else:
          _jacobian[n, m] = (_exp[n]/_denom)*(_exp[m]/_denom)

  return _jacobian

def sigmoid_derivative(z):
  return sigmoid(z)*(np.subtract(1,sigmoid(z)))

