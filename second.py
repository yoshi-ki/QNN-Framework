import numpy as np
import weakref


def as_array(x):
  if np.isscalar(x):
    return np.array(x)
  return x


class Variable:
  def __init__(self, data):
    # ndarrayとNone以外は受け付けないようにする
    if data is not None:
      if not isinstance(data, np.ndarray):
        raise TypeError('{} is not supported'.format(type(data)))
    self.data = data
    self.grad = None  # その変数をxとして、全体の出力をyとして、dy/dxが変数ごとに入る
    self.creator = None
    self.generation = 0

  def set_creator(self, func):
    self.creator = func
    self.generation = func.generation + 1

  def backward(self, retain_grad=False):
    if self.grad is None:
      self.grad = np.ones_like(self.data)

    funcs = []
    seen_set = set()

    def add_func(f):
      if f not in seen_set:
        seen_set.add(f)
        funcs.append(f)
        funcs.sort(key=lambda x: x.generation)

    add_func(self.creator)
    while funcs:
      # 今わかっていないnodeのgradを入力として、既知のgradのnodeを出力する関数を取得
      f = funcs.pop()
      gys = [output().grad for output in f.outputs]  # 関数の出力を取得
      # このbackward methodでは、dw/dxを計算して、dy/dwと掛け合わせることでdy/dxを計算してる(関数への入力がややこしいけど)
      gxs = f.backward(*gys)
      if not isinstance(gxs, tuple):
        gxs = (gxs,)
      for x, gx in zip(f.inputs, gxs):  # 関数の入力を取得
        if x.grad is None:
          x.grad = gx
        else:
          x.grad = x.grad + gx
        if x.creator is not None:
          add_func(x.creator)  # 一つ前の関数をリストに追加する
      if not retain_grad:
        for y in f.outputs:
          y().grad = None

  def cleargrad(self):
    self.grad = None


class Function:
  def __call__(self, *inputs):
    xs = [x.data for x in inputs]
    ys = self.forward(*xs)
    if not isinstance(ys, tuple):
      ys = (ys,)
    outputs = [Variable(as_array(y)) for y in ys]

    self.generation = max([x.generation for x in inputs])
    for output in outputs:
      output.set_creator(self)
    self.inputs = inputs
    self.outputs = [weakref.ref(output) for output in outputs]
    # 要素が一つのときは要素を返し、それ以外の時はリストを返す
    return outputs if len(outputs) > 1 else outputs[0]

  def forward(self, xs):
    raise NotImplementedError()

  def backward(self, gys):
    raise NotImplementedError()


class Square(Function):
  def forward(self, x):
    return x ** 2

  def backward(self, gy):
    x = self.inputs[0].data
    gx = 2 * x * gy
    return gx


class Exp(Function):
  def forward(self, x):
    return np.exp(x)

  def backward(self, gy):
    x = self.input.data
    gx = np.exp(x) * gy
    return gx


class Add(Function):
  def forward(self, x0, x1):
    y = x0 + x1
    return y

  def backward(self, gy):
    return gy, gy


def numerical_diff(f, x, eps=1e-4):
  # 中心差分での実装
  x0 = Variable(x.data - eps)
  x1 = Variable(x.data + eps)
  y0 = f(x0)
  y1 = f(x1)
  return (y1.data - y0.data) / (2 * eps)


def square(x):
  return Square()(x)


def exp(x):
  return Exp()(x)


def add(x0, x1):
  return Add()(x0, x1)


x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
t = add(x0, x1)
y = add(x0, t)
y.backward()
print(y.grad, t.grad)
print(x0.grad, x1.grad)
