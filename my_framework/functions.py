import numpy as np
from my_framework.core import Function, Variable
from my_framework.core import as_variable, as_array
from my_framework import utils


class Sin(Function):
  def forward(self, x):
    y = np.sin(x)
    return y

  def backward(self, gy):
    x, = self.inputs
    gx = gy * cos(x)
    return gx


def sin(x):
  return Sin()(x)


class Cos(Function):
  def forward(self, x):
    y = np.cos(x)
    return y

  def backward(self, gy):
    x, = self.inputs
    gx = gy * - sin(x)
    return gx


def cos(x):
  return Cos()(x)


class Tanh(Function):
  def forward(self, x):
    y = np.tanh(x)
    return y

  def backward(self, gy):
    y = self.outputs[0]()
    gx = gy * (1 - y * y)
    return gx


def tanh(x):
  return Tanh()(x)


class Exp(Function):
  def forward(self, x):
    y = np.exp(x)
    return y

  def backward(self, gy):
    y = self.outputs[0]()  # weakref
    gx = gy * y
    return gx


def exp(x):
  return Exp()(x)


class Log(Function):
  def forward(self, x):
    y = np.log(x)
    return y

  def backward(self, gy):
    x, = self.inputs
    gx = gy / x
    return gx


def log(x):
  return Log()(x)


class Reshape(Function):
  def __init__(self, shape):
    self.shape = shape

  def forward(self, x):
    self.x_shape = x.shape
    y = x.reshape(self.shape)
    return y

  def backward(self, gy):
    return reshape(gy, self.x_shape)


def reshape(x, shape):
  if x.shape == shape:
    return as_variable(x)
  return Reshape(shape)(x)


class Transpose(Function):
  def forward(self, x):
    y = np.transpose(x)
    return y

  def backward(self, gy):
    gx = transpose(gy)
    return gx


def transpose(x):
  return Transpose()(x)


class Sum(Function):
  def __init__(self, axis, keepdims):
    self.axis = axis
    self.keepdims = keepdims

  def forward(self, x):
    self.x_shape = x.shape
    y = x.sum(axis=self.axis, keepdims=self.keepdims)
    return y

  def backward(self, gy):
    gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
    gx = broadcast_to(gy, self.x_shape)
    return gx


def sum(x, axis=None, keepdims=False):
  return Sum(axis, keepdims)(x)


class BroadcastTo(Function):
  def __init__(self, shape):
    self.shape = shape

  def forward(self, x):
    self.x_shape = x.shape
    y = np.broadcast_to(x, self.shape)
    return y

  def backward(self, gy):
    gx = sum_to(gy, self.x_shape)
    return gx


def broadcast_to(x, shape):
  if x.shape == shape:
    return as_variable(x)
  return BroadcastTo(shape)(x)


class SumTo(Function):
  def __init__(self, shape):
    self.shape = shape

  def forward(self, x):
    self.x_shape = x.shape
    y = utils.sum_to(x, self.shape)
    return y

  def backward(self, gy):
    gx = broadcast_to(gy, self.x_shape)
    return gx


def sum_to(x, shape):
  if x.shape == shape:
    return as_variable(x)
  return SumTo(shape)(x)


class MatMul(Function):
  def forward(self, x, W):
    y = x.dot(W)
    return y

  def backward(self, gy):
    x, W = self.inputs
    gx = matmul(gy, W.T)
    gW = matmul(x.T, gy)
    return gx, gW


def matmul(x, W):
  return MatMul()(x, W)


class MeanSquaredError(Function):
  def forward(self, y, y_pred):
    diff = y - y_pred
    z = (diff ** 2).sum() / len(diff)
    return z

  def backward(self, gL):
    y, y_pred = self.inputs
    diff = y - y_pred
    gy = gL * diff * 2. / len(diff)
    gy_pred = -gy
    return gy, gy_pred


def mean_squared_error(y, y_pred):
  return MeanSquaredError()(y, y_pred)


class Linear(Function):
  def forward(self, x, W, b):
    y = x.dot(W)
    if b is not None:
      y += b
    return y

  def backward(self, gy):
    x, W, b = self.inputs
    gb = None if b.data is None else sum_to(gy, b.shape)
    gx = matmul(gy, W.T)
    gW = matmul(x.T, gy)
    return gx, gW, gb


def linear(x, W, b=None):
  return Linear()(x, W, b)


def linear_simple(x, W, b=None):
  x, W = as_variable(x), as_variable(W)
  t = matmul(x, W)
  if b is None:
    return t
  y = t + b
  t.data = None
  return y


# quantizeしたlinear class
# 出力データと重みをquantizeすればいい
# TODO: quantizeする
class Q_Linear(Function):
  def forward(self, x, W, b, bit_size):
    # W = #quantize W
    # binarize用
    # y = x.dot(W)
    # if b is not None:
    #   y += b
    # y = np.sign(y)

    # quantize用(ナイーブ)
    # bit_size = 1
    # self.bit_size = bit_size
    # y = x.dot(W)
    # W = np.round(W)
    # W = np.clip(W, - 2 ** (bit_size - 1), 2 ** (bit_size - 1) - 1)
    # if b is not None:
    #   y += b
    # y = np.round(y)
    # y = np.clip(y, - 2 ** (bit_size - 1), 2 ** (bit_size - 1) - 1)

    # quantize用(論文実装)
    # bit_size = 2
    self.bit_size = bit_size
    W_max = (W * np.sign(W)).max(axis=None, keepdims=True)
    self.W_max = W_max
    W = np.round(W * (2 ** (bit_size - 1)) / W_max)
    self.W_result = W
    W = np.clip(W, - 2 ** (bit_size - 1), 2 ** (bit_size - 1) - 1)
    y = x.dot(W)
    y_max = (y * np.sign(y)).max(axis=None, keepdims=True)
    self.y_max = y_max
    y = np.round(y * (2 ** (bit_size - 1)) / y_max)
    self.y_result = y
    y = np.clip(y, - 2 ** (bit_size - 1), 2 ** (bit_size - 1) - 1)

    return y

  def backward(self, gy):
    x, W, b, bit_size = self.inputs
    mask = (self.y_result >= - 2 ** (self.bit_size - 1)) * \
        (self.y_result <= 2 ** (self.bit_size - 1) - 1)
    gy = gy * mask * self.y_max
    # quantizeの効果をここでbackwardする
    gb = None if b.data is None else sum_to(gy, b.shape)
    gx = matmul(gy, W.T)
    gW = matmul(x.T, gy)
    mask = (self.W_result >= - 2 ** (self.bit_size - 1)) * \
        (self.W_result <= 2 ** (self.bit_size - 1) - 1)
    gW = gW * mask * self.W_max

    # quantize(ナイーブ)
    # x, W, b = self.inputs
    # # quantizeの効果をここでbackwardする
    # mask = ((x.dot(W)).data >= - 2 ** (self.bit_size - 1)) * \
    #     ((x.dot(W)).data <= 2 ** (self.bit_size - 1) - 1)
    # gy = gy * mask
    # gb = None if b.data is None else sum_to(gy, b.shape)
    # gx = matmul(gy, W.T)
    # gW = matmul(x.T, gy)
    # mask = (W.data >= - 2 ** (self.bit_size - 1)) * \
    #     (W.data <= 2 ** (self.bit_size - 1) - 1)
    # gW = gW * mask
    return gx, gW, gb


def q_linear(x, W, bit_size, b=None):
  return Q_Linear()(x, W, b, bit_size)


class B_Linear(Function):
  def forward(self, x, W, b):
    # binarize用
    W = np.sign(W)
    y = x.dot(W)
    if b is not None:
      y += b
    y = np.sign(y)

    return y

  def backward(self, gy):
    x, W, b = self.inputs
    gb = None if b.data is None else sum_to(gy, b.shape)
    gx = matmul(gy, W.T)
    gW = matmul(x.T, gy)
    return gx, gW, gb


def b_linear(x, W, b=None):
  return B_Linear()(x, W, b)


class Sigmoid(Function):
  def forward(self, x):
    y = 1 / (1 + np.exp(-x))
    return y

  def backward(self, gy):
    y = self.outputs[0]()
    gx = gy * y * (1 - y)
    return gx


def sigmoid(x):
  return Sigmoid()(x)


class GetItem(Function):
  def __init__(self, slices):
    self.slices = slices

  def forward(self, x):
    y = x[self.slices]
    return y

  def backward(self, gy):
    x, = self.inputs
    f = GetItemGrad(self.slices, x.shape)
    return f(gy)


def get_item(x, slices):
  f = GetItem(slices)
  return f(x)


class GetItemGrad(Function):
  def __init__(self, slices, in_shape):
    self.slices = slices
    self.in_shape = in_shape

  def forward(self, gy):
    gx = np.zeros(self.in_shape)
    np.add.at(gx, self.slices, gy)
    return gx

  def backward(self, ggx):
    return get_item(ggx, self.slices)


class Softmax(Function):
  def __init__(self, axis=1):
    self.axis = axis

  def forward(self, x):
    y = x - x.max(axis=self.axis, keepdims=True)
    # xのtypeは<class 'numpy.ndarray'>
    y = as_variable(y)
    y = exp(y)
    sum_y = y.sum(axis=self.axis, keepdims=True)
    return y / sum_y

  def backward(self, gy):
    y = self.outputs[0]()
    gx = y * gy
    sumdx = gx.sum(axis=self.axis, keepdims=True)
    gx -= y * sumdx
    return gx


def softmax(x, axis=1):
  return Softmax(axis)(x)


def softmax_simple(x, axis=1):
  x = x - x.max(axis=axis, keepdims=True)
  # x = x
  x = as_variable(x)
  y = exp(x)
  sum_y = sum(y, axis=axis, keepdims=True)
  return y / sum_y


def softmax_cross_entropy_simple(x, t):
  x, t = as_variable(x), as_variable(t)
  N = x.shape[0]
  p = softmax_simple(x)
  p = clip(p, 1e-15, 1.0)  # To avoid log(0)
  log_p = log(p)
  tlog_p = log_p[np.arange(N), t.data]
  y = -1 * sum(tlog_p) / N
  return y


class Clip(Function):
  def __init__(self, x_min, x_max):
    self.x_min = x_min
    self.x_max = x_max

  def forward(self, x):
    y = np.clip(x, self.x_min, self.x_max)
    return y

  def backward(self, gy):
    x, = self.inputs
    mask = (x.data >= self.x_min) * (x.data <= self.x_max)
    gx = gy * mask
    return gx


def clip(x, x_min, x_max):
  return Clip(x_min, x_max)(x)


class Max(Function):
  def __init__(self, axis=None, keepdims=False):
    self.axis = axis
    self.keepdims = keepdims

  def forward(self, x):
    y = x.max(axis=self.axis, keepdims=self.keepdims)
    return y

  def backward(self, gy):
    x = self.inputs[0]
    y = self.outputs[0]()  # weakref

    shape = utils.max_backward_shape(x, self.axis)
    gy = reshape(gy, shape)
    y = reshape(y, shape)
    cond = (x.data == y.data)
    gy = broadcast_to(gy, cond.shape)
    return gy * cond


class Min(Max):
  def forward(self, x):
    y = x.min(axis=self.axis, keepdims=self.keepdims)
    return y


def max(x, axis=None, keepdims=False):
  return Max(axis, keepdims)(x)


def min(x, axis=None, keepdims=False):
  return Min(axis, keepdims)(x)


def accuracy(y, t):
  y, t = as_variable(y), as_variable(t)
  pred = y.data.argmax(axis=1).reshape(t.shape)
  result = (pred == t.data)
  acc = result.mean()
  return Variable(as_array(acc))


# from my_framework.core import add
# from my_framework.core import sub
# from my_framework.core import rsub
# from my_framework.core import mul
# from my_framework.core import div
# from my_framework.core import neg
# from my_framework.core import pow
