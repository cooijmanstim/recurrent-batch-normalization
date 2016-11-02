import sys, numpy as np
import cPickle as pkl
import matplotlib.pyplot as plt
from matplotlib import cm

friendly_labels = {
  "Unidir": "BN-e unidir tweaked",
  "Bidir": "BN-e bidir tweaked",
}

paths = """
LSTM:full/baseline.npz.pkl
Bidir:full/120_uni_bs64_lr8e-4_use_dq_sims1_use_desc_skip_c_g1_sequencewise/stats_bn-bidir-dropout.npz.pkl
""".split()

#Unidir:full/bn_280_bi_bs64_lr8e-4_use_dq_sims1_use_desc_skip_c_g1_sequensewisenorm/stats_bn-bidir-dropout.npz.pkl

instances = []
for i, path in enumerate(paths):
    try:
        label, path = path.split(":")
    except ValueError:
        label = i
    label = friendly_labels.get(label, label)
    print(label, path)
    instances.append(dict(label=label,
                          path=path,
                          data=pkl.load(open(path))))

colors = "r b g goldenrod purple".split()
#colors = cm.rainbow(np.linspace(0, 1, len(instances)))

channel_labels = dict(train_err_ave="train",
                      valid_errs="valid")

import matplotlib
matplotlib.rcParams.update({"font.size": 18})
plt.figure()
for channel_name, kwargs in [
        ("train_err_ave", dict(linestyle="dotted")),
        ("valid_errs",    dict(linestyle="solid"))]:
    for color, instance in zip(colors, instances):
        label = "%s %s" % (instance["label"], channel_labels[channel_name])
        x = np.asarray(instance["data"][channel_name])
        minimum = min(x)
        print label, "min", minimum
        plt.plot(x, label=label, color=color, linewidth=2, **kwargs)
        if "valid" in channel_name:
          plt.axhline(y=minimum, xmin=0.98, xmax=1, linewidth=2, color=color)

plt.legend(prop=dict(size=14))
plt.xlim((0, 400))
plt.ylabel("error rate")
plt.xlabel("training steps (thousands)")

if False:
  plt.figure()
  for channel_name, kwargs in [
          ("train_cost_ave", dict(linestyle="dotted")),
          ("valid_costs",    dict(linestyle="solid"))]:
      for color, instance in zip(colors, instances):
          label = "%s %s" % (instance["label"], channel_name)
          plt.plot(instance["data"][channel_name], label=label, color=color, linewidth=3, **kwargs)
  plt.legend(prop=dict(size=14))
  plt.xlim((0, 400))
  plt.ylabel("cost")
  plt.xlabel("training steps")

plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(800 / fig.dpi, 600 / fig.dpi)
plt.savefig("attr_full_valid.pdf", bbox_inches="tight")
#plt.show()
