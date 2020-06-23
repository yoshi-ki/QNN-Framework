if '__file__' in globals():
  import os
  import sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from my_framework import Variable, Model
import my_framework.functions as F
import my_framework.layers as L

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)


class TwoLayerNet(Model):
  def __init__(self, hidden_size, out_size):
    super().__init__()
    self.l1 = L.Linear(hidden_size)
    self.l2 = L.Linear(out_size)

  def forward(self, x):
    y = F.sigmoid(self.l1(x))
    y = self.l2(y)
    return y


lr = 0.2
iters = 10000
hidden_size = 10

model = TwoLayerNet(hidden_size, 1)

for i in range(iters):
  y_pred = model(x)
  loss = F.mean_squared_error(y, y_pred)

  model.cleargrads()
  loss.backward()

  for p in model.params():
    p.data -= lr * p.grad.data
  if i % 1000 == 0:
    print(loss)
