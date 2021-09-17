# Defined in file: ./chapter_linear-networks/linear-regression.md 3.1线性回归
import time
from IPython import display
import numpy as np
import matplotlib.pyplot as plt
import paddle
from paddle.io import TensorDataset, DataLoader
from paddle.vision.datasets import FashionMNIST
import paddle.vision.transforms as transforms


class Timer:
  """Record multiple running times."""

  def __init__(self):
    self.times = []
    self.start()

  def start(self):
    """Start the timer."""
    self.tik = time.time()

  def stop(self):
    """Stop the timer and record the time in a list."""
    self.times.append(time.time() - self.tik)
    return self.times[-1]

  def avg(self):
    """Return the average time."""
    return sum(self.times) / len(self.times)

  def sum(self):
    """Return the sum of time."""
    return sum(self.times)

  def cumsum(self):
    """Return the accumulated time."""
    return np.array(self.times).cumsum().tolist()


# Defined in file: ./chapter_linear-networks/linear-regression-scratch.md  3.2线性回归从零开始实现
def synthetic_data(w, b, num_examples):
  """Generate y = Xw + b + noise."""
  X = np.random.normal(0, 1, (num_examples, len(w)))
  y = np.dot(X, w) + b
  y += np.random.normal(0, 0.01, y.shape)
  return X, y.reshape((-1, 1))


# Defined in file: ./chapter_linear-networks/linear-regression-scratch.md  3.2.4. 定义模型
def linreg(X, w, b):
  """The linear regression model."""
  return np.dot(X, w) + b


# Defined in file: ./chapter_linear-networks/linear-regression-scratch.md  3.2.5. 定义损失函数
def squared_loss(y_hat, y):
  """Squared loss."""
  return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# Defined in file: ./chapter_linear-networks/linear-regression-scratch.md  3.2.6. 定义优化算法
def sgd(params, lr, batch_size):
  """Minibatch stochastic gradient descent."""
  for param in params:
    param[:] = param - lr * param.grad / batch_size


# Defined in file: ./chapter_linear-networks/linear-regression-concise.md  3.3.2. 读取数据集
def load_array(data_arrays, batch_size, is_train=True):
  """Construct a PyTorch data iterator."""
  # dataset = data.TensorDataset(*data_arrays)
  dataset = TensorDataset(*data_arrays)
  # return data.DataLoader(dataset, batch_size, shuffle=is_train)
  return DataLoader(dataset, batch_size, shuffle=is_train)


# Defined in file: ./chapter_linear-networks/image-classification-dataset.md  3.5.1. 读取数据集
def get_fashion_mnist_labels(labels):
  """Return text labels for the Fashion-MNIST dataset."""
  text_labels = [
    't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
    'sneaker', 'bag', 'ankle boot']
  return [text_labels[int(i)] for i in labels]


# Defined in file: ./chapter_linear-networks/image-classification-dataset.md  3.5.1. 读取数据集
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
  """Plot a list of images."""
  figsize = (num_cols * scale, num_rows * scale)
  _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
  axes = axes.flatten()
  for i, (ax, img) in enumerate(zip(axes, imgs)):
      if paddle.is_tensor(img):
          # 图片张量
          ax.imshow(img.numpy())
      else:
          # PIL图片
          ax.imshow(img)
      ax.axes.get_xaxis().set_visible(False)
      ax.axes.get_yaxis().set_visible(False)
      if titles:
          ax.set_title(titles[i])
  return axes


# Defined in file: ./chapter_linear-networks/image-classification-dataset.md 3.5.2. 读取小批量
def get_dataloader_workers():
  """Use 4 processes to read the data."""
  return 4


# Defined in file: ./chapter_linear-networks/image-classification-dataset.md  3.5.3. 整合所有组件
import paddle.vision.transforms as T
def load_data_fashion_mnist(batch_size, resize=None):
  """Download the Fashion-MNIST dataset and then load it into memory."""
  trans = [T.Compose([T.Normalize(mean=[127.5],std=[127.5],data_format='CHW')])]
  if resize:
      trans.insert(0, transforms.Resize(resize))
  trans = transforms.Compose(trans)
  mnist_train = torchvision.datasets.FashionMNIST(
      root="../data", train=True, transform=trans, download=True)
  mnist_test = torchvision.datasets.FashionMNIST(
      root="../data", train=False, transform=trans, download=True)
  return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                          num_workers=get_dataloader_workers()),
          data.DataLoader(mnist_test, batch_size, shuffle=False,
                          num_workers=get_dataloader_workers()))


# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md  3.6.5. 分类准确率
def accuracy(y_hat, y):
  """Compute the number of correct predictions."""
  if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
    y_hat = y_hat.argmax(axis=1)
  cmp = y_hat.astype(y.dtype) == y
  return float(cmp.astype(y.dtype).sum())


# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md  3.6.5. 分类准确率
def evaluate_accuracy(net, data_iter):
  """Compute the accuracy for a model on a dataset."""
  metric = Accumulator(2)  # 正确预测数、预测总数
  for X, y in data_iter:
    metric.add(accuracy(net(X), y), y.size)
  return metric[0] / metric[1]


# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md  3.6.5. 分类准确率
class Accumulator:
  """For accumulating sums over `n` variables."""

  def __init__(self, n):
    self.data = [0.0] * n

  def add(self, *args):
    self.data = [a + float(b) for a, b in zip(self.data, args)]

  def reset(self):
    self.data = [0.0] * len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]


# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md  3.6.6. 训练
def train_epoch_ch3(net, train_iter, loss, updater):  # @save
  """训练模型一个迭代周期（定义见第3章）。"""
  # 将模型设置为训练模式
  if isinstance(net, paddle.nn.Layer):
    net.train()
  # 训练损失总和、训练准确度总和、样本数
  metric = Accumulator(3)
  for X, y in train_iter:
    # 计算梯度并更新参数
    y_hat = net(X)
    l = loss(y_hat, y)
    if isinstance(updater,paddle.optimizer.Optimizer):
      # 使用内置的优化器和损失函数
      updater.zero_grad()
      l.backward()
      updater.step()
      metric.add(
        float(l) * len(y), accuracy(y_hat, y),
        y.size().numel())
    else:
      # 使用定制的优化器和损失函数
      l.sum().backward()
      updater(X.shape[0])
      metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
  # 返回训练损失和训练准确率
  return metric[0] / metric[2], metric[1] / metric[2]


# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md  3.6.6. 训练
class Animator:
  """For plotting data in animation."""

  def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
               ylim=None, xscale='linear', yscale='linear',
               fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
               figsize=(3.5, 2.5)):
    # Incrementally plot multiple lines
    if legend is None:
      legend = []
    d2l.use_svg_display()
    self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows * ncols == 1:
      self.axes = [self.axes, ]
    # Use a lambda function to capture arguments
    self.config_axes = lambda: plt.set_axes(self.axes[
                                              0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    self.X, self.Y, self.fmts = None, None, fmts

  def add(self, x, y):
    # Add multiple data points into the figure
    if not hasattr(y, "__len__"):
      y = [y]
    n = len(y)
    if not hasattr(x, "__len__"):
      x = [x] * n
    if not self.X:
      self.X = [[] for _ in range(n)]
    if not self.Y:
      self.Y = [[] for _ in range(n)]
    for i, (a, b) in enumerate(zip(x, y)):
      if a is not None and b is not None:
        self.X[i].append(a)
        self.Y[i].append(b)
    self.axes[0].cla()
    for x, y, fmt in zip(self.X, self.Y, self.fmts):
      self.axes[0].plot(x, y, fmt)
    self.config_axes()
    display.display(self.fig)
    display.clear_output(wait=True)


# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md  3.6.6. 训练
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
  """Train a model (defined in Chapter 3)."""
  animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                      legend=['train loss', 'train acc', 'test acc'])
  for epoch in range(num_epochs):
    train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
    test_acc = evaluate_accuracy(net, test_iter)
    animator.add(epoch + 1, train_metrics + (test_acc,))
  train_loss, train_acc = train_metrics
  assert train_loss < 0.5, train_loss
  assert train_acc <= 1 and train_acc > 0.7, train_acc
  assert test_acc <= 1 and test_acc > 0.7, test_acc


# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md  3.6.7. 预测
def predict_ch3(net, test_iter, n=6):
  """Predict labels (defined in Chapter 3)."""
  for X, y in test_iter:
    break
  trues = d2l.get_fashion_mnist_labels(y)
  preds = d2l.get_fashion_mnist_labels(paddle.argmax(net(X), axis=1))
  titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
  d2l.show_images(paddle.reshape(X[0:n], (n, 28, 28)), 1, n,
                  titles=titles[0:n])
