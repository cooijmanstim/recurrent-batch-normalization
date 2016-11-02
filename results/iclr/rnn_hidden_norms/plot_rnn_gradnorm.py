import sys, numpy as np
from blocks.serialization import load
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import OrderedDict
import matplotlib
matplotlib.rcParams.update({"font.size": 20})

def split_path(pathlike):
    i, pathlike = pathlike
    try:
        name, path = pathlike.split(":")
    except (ValueError, AttributeError):
        name, path = i, pathlike
    print("%s: %s" % (name, path))
    return name, path

def load_instance(pathlike):
    name, path = split_path(pathlike)
    return load_named_instance(name, path)

def load_named_instance(name, path):
    with open(path, "rb") as file:
        hiddens = load(file)
    return dict(name=name, path=path, hiddens=hiddens)

def load_named_instance(name, path):
  print(name, path)
  npz_path = path.replace(".pkl", ".npz")
  try:
    hiddens = np.load(npz_path)
  except:
    with open(path, "rb") as file:
      hiddens = load(file)
    np.savez_compressed(npz_path, **hiddens)
  return dict(name=name, path=path, hiddens=hiddens)

#paths = sys.argv[1:]
#instances = list(map(load_instance, enumerate(paths)))

paths = OrderedDict([
("gamma=0.10", "rnn_gamma0.10/hiddens_0.pkl"),
("gamma=0.20", "rnn_gamma0.20/hiddens_0.pkl"),
("gamma=0.30", "rnn_gamma0.30/hiddens_0.pkl"),
("gamma=0.40", "rnn_gamma0.40/hiddens_0.pkl"),
("gamma=0.50", "rnn_gamma0.50/hiddens_0.pkl"),
("gamma=0.60", "rnn_gamma0.60/hiddens_0.pkl"),
("gamma=0.70", "rnn_gamma0.70/hiddens_0.pkl"),
("gamma=0.80", "rnn_gamma0.80/hiddens_0.pkl"),
("gamma=0.90", "rnn_gamma0.90/hiddens_0.pkl"),
("gamma=1.00", "rnn_gamma1.0/hiddens_0.pkl"),
])
instances = [load_named_instance(k, v) for k, v in paths.items()]

colors = cm.viridis(np.linspace(0.1, 0.9, len(instances)))
linestyles = "- - - - - - - - - - - - - - - -".split()
assert len(linestyles) >= len(instances)

plt.figure()
allnorms = []
for instance, color, linestyle in zip(instances, colors, linestyles):
    # expected gradient norm over time (expectation over data)
    norms = np.sqrt((instance["hiddens"]["h_grad"] ** 2).sum(axis=2)).mean(axis=1)
    plt.plot(norms, label=instance["name"], color=color, linewidth="3", linestyle=linestyle)
    allnorms.extend(norms)

plt.title("RNN gradient propagation")
plt.yscale("log")
plt.ylim(ymax=1)
plt.xlabel("t")
plt.ylabel("||dloss/dh_t||_2")
plt.legend(loc="lower left", prop=dict(size=16))

axis = plt.gca()
yticks = axis.get_yticks()
# ticks are on odd powers for some reason
yticks *= 10
# ticks exceed range for some reason
yticks = yticks[yticks <= axis.get_ylim()[1]]
axis.set_yticks(yticks)

#import pdb; pdb.set_trace()
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(800 / fig.dpi, 600 / fig.dpi)
plt.savefig("rnn_grad_prop.pdf", bbox_inches="tight")
#plt.show()
