import os, sys, pickle, zipfile, math
from collections import OrderedDict
from itertools import starmap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from blocks.serialization import load
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

matplotlib.rcParams.update({"font.size": 10})

def get(name, path, channel_name):
  print(name, path)
  npz_path = path.replace(".pkl", "_%s.npz" % channel_name)
  try:
    data = np.load(npz_path)
    t = np.array(data["t"])
    v = np.array(data["v"])
  except:
    with open(path, "rb") as file:
      log = load(file)
    tv = np.array([[t, v[channel_name]]
                   for t, v in log.items()
                   if v and channel_name in v])
    log = None
    t = tv[:, 0]
    v = tv[:, 1]
    np.savez_compressed(npz_path, t=t, v=v)
  return dict(name=name, path=path, t=t, v=v)

def plot_mnist(fold, ax):
  instances = [
    dict(name="gamma 0.10", path="pmnist/gamma_1e-1/log.pkl"),
    dict(name="gamma 0.30", path="pmnist/gamma_3e-1/log.pkl"),
    dict(name="gamma 0.50", path="pmnist/gamma_5e-1/log.pkl"),
    dict(name="gamma 0.70", path="pmnist/gamma_7e-1/log.pkl"),
    dict(name="gamma 1.00", path="pmnist/gamma_1/log.pkl"),
  ]
  channel_name = "%s_training_cross_entropy" % fold
  instances = [get(channel_name=channel_name, **instance)
               for instance in instances]
  plot(instances, ax=ax)
  ax.set_ylabel("cross entropy")
  ax.set_xlabel("training steps")
  ax.set_ylim(dict(train=[0.0, 2.5],
                   valid=[0.0, 2.5])[fold])
  ax.set_xlim(dict(train=[0, 50000],
                   valid=[0, 50000])[fold])
  ax.set_title("Permuted MNIST %s" % fold)

def plot_ptb(fold, ax):
  instances = [
    #dict(name="LSTM",          path="ptb/baseline/log.pkl"),
    #dict(name="gamma 0.01", path="ptb/batchnorm-ig0.01/log.pkl"),
    dict(name="gamma 0.10", path="ptb/batchnorm-ig0.10/log.pkl"),
    #dict(name="gamma 0.20", path="ptb/batchnorm-ig0.20/log.pkl"),
    dict(name="gamma 0.30", path="ptb/batchnorm-ig0.30/log.pkl"),
    #dict(name="gamma 0.40", path="ptb/batchnorm-ig0.40/log.pkl"),
    dict(name="gamma 0.50", path="ptb/batchnorm-ig0.50/log.pkl"),
    #dict(name="gamma 0.60", path="ptb/batchnorm-ig0.60/log.pkl"),
    dict(name="gamma 0.70", path="ptb/batchnorm-ig0.70/log.pkl"),
    #dict(name="gamma 0.80", path="ptb/batchnorm-ig0.80/log.pkl"),
    #dict(name="gamma 0.90", path="ptb/batchnorm-ig0.90/log.pkl"),
    dict(name="gamma 1.00", path="ptb/batchnorm-ig1.00/log.pkl"),
  ]
  channel_name = "%s_training_cross_entropy" % fold
  instances = [get(channel_name=channel_name, **instance)
               for instance in instances]

  for instance in instances:
    # nats to bits
    instance["t"] /= math.log(2)
  
  plot(instances, ax=ax)
  ax.set_ylabel("bits per character")
  ax.set_xlabel("training steps")
  ax.set_ylim(dict(train=[0.8, 1.1],
                   valid=[1.0, 1.1])[fold])
  ax.set_xlim(dict(train=[0, 19000],
                   valid=[0, 19000])[fold])
  ax.set_title("PTB %s" % fold)

def plot(instances, ax=None):
  colors = cm.viridis(np.linspace(.2, .8, len(instances)))
  for color, instance in zip(colors, instances):
    ax.plot(instance["t"], instance["v"], label=instance["name"],
            c=color, linewidth=1)
  ax.legend(prop=dict(size=10))

if __name__ == "__main__":
  fig, axes = plt.subplots(2, 2)
  plot_mnist(fold="train", ax=axes[0][0])
  plot_mnist(fold="valid", ax=axes[0][1])
  plot_ptb(fold="train", ax=axes[1][0])
  plot_ptb(fold="valid", ax=axes[1][1])
  plt.tight_layout()
  #plt.show()
  fig.set_size_inches(800 / fig.dpi, 600 / fig.dpi)
  plt.savefig("gammas.pdf", bbox_inches="tight")
