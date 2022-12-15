import numpy as np


def binary(z, thresh=0.5):
  return [0 if z_i<thresh else 1 for z_i in z]

def relu(z):
  return [np.max([0, z_i]) for z_i in z]

def leaky_relu(z, alpha=0.1):
  return [np.max([alpha*z_i, z_i]) for z_i in z]

def elu(z, alpha=0.1):
  return [np.max([alpha*(np.exp(z_i)-1), z_i]) for z_i in z]

def softmax(z):
  sum = 0
  for z_i in z:
    sum += np.exp(z_i)

  return [np.divide(np.exp(z_i), sum) for z_i in z]

def sigmoid(z):
  return [np.divide(1, (1+np.exp(-z)))]

def binary_derivative(z):
  return [0]

def relu_derivative(z):
  return [0 if z<0 else 1]

def leaky_relu_derivative(z, alpha=0.1):
  return [alpha if z<0 else 1]

def elu_derivative(z, alpha=0.1):
  return [alpha if z<0 else 1]

def softmax_derivative():
  pass

def sigmoid_derivative(z):
  return sigmoid(z)*([1]-sigmoid(z))

