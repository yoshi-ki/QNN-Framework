if '__file__' in globals():
  import os
  import sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import my_framework
from my_framework import optimizers, DataLoader
import my_framework.functions as F
from my_framework.models import MLP
import matplotlib.pyplot as plt


# def f(x):
#   x = x.flatten()
#   x = x.astype(np.float32)
#   x /= 255
#   return x


# train_set = my_framework.datasets.MNIST(train=True, transform=f)
# test_set = my_framework.datasets.MNIST(train=False, transform=f)
max_epoch = 200
batch_size = 100
hidden_size = 1000
bit_size = 1

train_set = my_framework.datasets.MNIST(train=True)
test_set = my_framework.datasets.MNIST(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)


model = MLP((hidden_size, 10))
optimizer = optimizers.SGD().setup(model)

train_acc = np.zeros(max_epoch)
test_acc = np.zeros(max_epoch)
train_loss = np.zeros(max_epoch)
test_loss = np.zeros(max_epoch)

for epoch in range(max_epoch):
  sum_loss, sum_acc = 0, 0

  for x, t in train_loader:
    y = model(x)
    loss = F.softmax_cross_entropy_simple(y, t)
    acc = F.accuracy(y, t)
    model.cleargrads()
    loss.backward()
    optimizer.update()

    sum_loss += float(loss.data) * len(t)
    sum_acc += float(acc.data) * len(t)

  print('epoch: {}'.format(epoch + 1))
  print('train loss: {:.4f}, accuracy: {:.4f}'.format(
      sum_loss / len(train_set), sum_acc / len(train_set)))

  # グラフ描画用
  train_acc[epoch] = sum_acc / len(train_set)
  train_loss[epoch] = sum_loss / len(train_set)

  sum_loss, sum_acc = 0, 0
  with my_framework.no_grad():
    for x, t in test_loader:
      y = model(x)
      loss = F.softmax_cross_entropy_simple(y, t)
      acc = F.accuracy(y, t)
      sum_loss += float(loss.data) * len(t)
      sum_acc += float(acc.data) * len(t)

  print('test loss: {:.4f}, accuracy: {:.4f}'.format(
      sum_loss / len(test_set), sum_acc / len(test_set)))

  # グラフ描画用
  test_acc[epoch] = sum_acc / len(test_set)
  test_loss[epoch] = sum_loss / len(test_set)


plt.figure()
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.plot(np.arange(max_epoch), train_acc, label="train", color="blue")
plt.plot(np.arange(max_epoch), test_acc, label="test", color="orange")
plt.legend()
plt.savefig("mnist_result_acc" + str(bit_size) + ".png")

plt.figure()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(np.arange(max_epoch), train_loss, label="train", color="blue")
plt.plot(np.arange(max_epoch), test_loss, label="test", color="orange")
plt.legend()
plt.savefig("mnist_result_loss" + str(bit_size) + ".png")
