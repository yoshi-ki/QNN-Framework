from my_framework.layers import Layer
from my_framework import utils
import my_framework.functions as F
import my_framework.layers as L


class Model(Layer):
  def plot(self, *inputs, to_file='model.png'):
    y = self.forward(*inputs)
    return utils.plot_dot_graph(y, verbose=True, to_file=to_file)


class MLP(Model):
  # MLPは全結合NNの抽象クラス
  def __init__(self, fc_output_sizes, activation=F.sigmoid):
    super().__init__()
    self.activation = activation
    self.layers = []

    for i, out_size in enumerate(fc_output_sizes):
      layer = L.Linear(out_size)
      setattr(self, 'l' + str(i), layer)
      self.layers.append(layer)

  def forward(self, x):
    for l in self.layers[:-1]:
      x = self.activation(l(x))
    return self.layers[-1](x)

# quantizeされた全結合ニューラルネットワーク


class Q_MLP(Model):
  # MLPは全結合NNの抽象クラス
  def __init__(self, fc_output_sizes, bit_size, activation=F.sigmoid):
    super().__init__()
    self.activation = activation
    self.layers = []

    for i, out_size in enumerate(fc_output_sizes):
      # 最終層はquantizeしない
      if i == (len(fc_output_sizes) - 1):
        layer = L.Linear(out_size)
      else:
        # 入力を受け取って、計算を行ってquantizeを行う
        layer = L.Q_Linear(out_size, bit_size)
      setattr(self, 'l' + str(i), layer)
      self.layers.append(layer)

  def forward(self, x):
    for l in self.layers[:-1]:
      x = self.activation(l(x))
    return self.layers[-1](x)
