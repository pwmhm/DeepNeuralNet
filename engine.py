import numpy as np
import activation as activ

class Layer():
  def __init__(self, activation, input, output, initialize):
    self.activate = activation
    self.derivative = getattr(activ, f"{activation.__name__}_derivative")
    self.weights = initialize((input, output))

  def forward(self, input):
    retval = np.matmul(input, self.weights)
    return self.activate(retval)

  def backward(self, input):
    retval = np.matmul(input, self.weights)
    return self.derivative(retval)

class Model():
  def __init__(self, eval=False):
    self.backbone = []
    self.eval = eval
    self.lr = 0
    self.record = []

  def add(self, layer):
    self.backbone.append(layer)

  def forward(self, input):
    retval = []
    for step, hlayers in enumerate(self.backbone):
      if step == 0:
        retval = hlayers.forward(input)
      else:
        retval = hlayers.forward(retval)
      if self.eval:
        self.record.append(retval)
    return retval

  def train(self):
    pass

  def loss(self, *args):
    """
    Placeholder for the loss gradient
    """
    pass

  def backward(self, input):
    temp_name = []
    for layer in range(len(self.backbone)-1, -1, -1):
      if layer == len(self.backbone)-1:
        gradient = self.backbone[layer].backward(self.record[layer-1]) * self.loss(self.record[layer])
        temp_name.append(gradient)
      else:
        gradient = self.backbone[layer].backward(self.record[layer-1]) * self.backbone[layer].weights
        gradient = np.matmul(gradient, temp_name[-1])
        temp_name.append(gradient)

        self.backbone[layer].weights = self.backbone[layer].weights - self.lr*gradient
