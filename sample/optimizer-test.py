if '__file__' in globals():
  import os
  import sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from my_framework import Variable, Model, optimizers
import my_framework.functions as F
import my_framework.layers as L
from my_framework.models import MLP


np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)


lr = 0.2
iters = 10000
hidden_size = 10

model = MLP((hidden_size, 1))
optimizer = optimizers.SGD(lr)
optimizer.setup(model)


for i in range(iters):
  y_pred = model(x)
  loss = F.mean_squared_error(y, y_pred)

  model.cleargrads()
  loss.backward()

  optimizer.update()
  if i % 1000 == 0:
    print(loss)
