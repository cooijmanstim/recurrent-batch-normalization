import sys, numpy as np
import cPickle as pkl
import matplotlib.pyplot as plt
from matplotlib import cm

paths = sys.argv[1:]
instances = []
for i, path in enumerate(paths):
    try:
        label, path = path.split(":")
    except ValueError:
        label = i
    print(label, path)
    instances.append(dict(label=label,
                          path=path,
                          data=pkl.load(open(path))))

import pdb; pdb.set_trace()

colors = "r b g purple maroon darkslategray darkolivegreen orangered".split()
colors = cm.rainbow(np.linspace(0, 1, len(instances)))

channel_labels = dict(train_err_ave="train",
                      valid_errs="valid")

plt.figure()
for channel_name, kwargs in [
        ("train_err_ave", dict(linestyle="dotted")),
        ("valid_errs",    dict(linestyle="solid"))]:
    for color, instance in zip(colors, instances):
        label = "%s %s" % (instance["label"], channel_labels[channel_name])
        plt.plot(np.asarray(instance["data"][channel_name]), label=label, color=color, linewidth=3, **kwargs)
plt.legend()
plt.xlim((0, 800))
plt.ylabel("error rate")
plt.xlabel("training steps (thousands)")

plt.figure()
for channel_name, kwargs in [
        ("train_cost_ave", dict(linestyle="dotted")),
        ("valid_costs",    dict(linestyle="solid"))]:
    for color, instance in zip(colors, instances):
        label = "%s %s" % (instance["label"], channel_name)
        plt.plot(instance["data"][channel_name], label=label, color=color, linewidth=3, **kwargs)
plt.legend()
plt.xlim((0, 800))
plt.ylabel("error rate")
plt.xlabel("training steps")

plt.show()

