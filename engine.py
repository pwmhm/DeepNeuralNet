import numpy as np
import activation as activ

class Layer():
  """
  Hidden layers that houses the activation functions. A layer is assumed to have 
  the same type of activation, no matter the amount.
  """
  def __init__(self, activation, input, output, initialize):
    self.activate = activation
    self.derivative = getattr(activ, f"{activation.__name__}_derivative")
    self.weights = initialize((input, output))
    self.identity = f"({activation.__name__}, {input}, {output})"

  def forward(self, input):
    """
    Does forward propagation given an array of input.
    """
    retval = np.matmul(input, self.weights)
    return self.activate(retval)

  def backward(self, input):
    """
    Does forward propagation TO the derivative of the activation function (Necessary for backprop).
    """
    retval = np.matmul(input, self.weights)
    return self.derivative(retval)

class Model():
  def __init__(self, eval=True):
    self.backbone = []
    self.eval = eval
    self.record = []

  def net_print(self):
    retval = f"Net:(\n"
    for layers in self.backbone:
      retval += f"{layers.identity},\n"
    retval += ")"
    return retval

  def add(self, layer):
    self.backbone.append(layer)

  def forward(self, input):
    retval = []
    if not self.eval:
      self.record = [input]
    for step, hlayers in enumerate(self.backbone):
      if step == 0:
        retval = hlayers.forward(input)
      else:
        retval = hlayers.forward(retval)
      if self.eval:
        self.record.append(retval)
    return retval

  def train(self, dataset, learning_rate, loss, metric, epochs):
    """
    This should be the default training schema.
    """
    self.lr = learning_rate
    self.data = dataset
    self.loss_module = loss
    self.metric = metric
    self.epochs = epochs
    self.eval = True

    # repeat for x epochs
    # TODO mini-batch and full-batch implementation
    for i in range(self.epochs):
      # iterate over training data
      for input, gt in dataset:
        # calculate predicted output (stored in self.record)
        self.forward(input)

        # calculate loss and derivatives
        self.loss_module(input, gt, self.eval, False)

        # backpropagate
        self.backward()

  def backward(self):
    temp_name = []
    for layer in range(len(self.backbone)-1, -1, -1):
      if layer == len(self.backbone)-1:
        # if it's the last layer, gradient calculation should be different
        gradient = self.backbone[layer].backward(self.record[layer-1]) * np.mean(self.loss_module.record)
        temp_name.append(gradient)
      else:
        gradient = self.backbone[layer].backward(self.record[layer-1]) * self.backbone[layer].weights
        gradient = np.matmul(gradient, temp_name[-1])
        temp_name.append(gradient)

      # update the weights
      self.backbone[layer].weights = self.backbone[layer].weights - self.lr*gradient
