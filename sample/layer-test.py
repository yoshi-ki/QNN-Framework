if '__file__' in globals():
  import os
  import sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from my_framework import Variable
from my_framework.utils import plot_dot_graph
import my_framework.functions as F
import my_framework.layers as L

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# I, H, O = 1, 10, 1
# W1 = Variable(0.01 * np.random.randn(I, H))
# b1 = Variable(np.zeros(H))
# W2 = Variable(0.01 * np.random.randn(H, O))
# b2 = Variable(np.zeros(O))

l1 = L.Linear(10)
l2 = L.Linear(1)


def predict(x):
  y = l1(x)
  y = F.sigmoid(y)
  y = l2(y)
  return y


lr = 0.2
iters = 10000

for i in range(iters):
  y_pred = predict(x)
  loss = F.mean_squared_error(y, y_pred)

  l1.cleargrads()
  l2.cleargrads()
  loss.backward()

  for l in [l1, l2]:
    for p in l.params():
      p.data -= lr * p.grad.data
  if i % 1000 == 0:
    print(loss)
