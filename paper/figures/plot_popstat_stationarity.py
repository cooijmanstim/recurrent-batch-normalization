import cPickle, numpy as np

pkl = cPickle.load(open("/data/lisatmp3/cooijmat/run/batchnorm/ptb/repeatpopstat_independentbatchstat/checkpoint.zip_popstat_results.pkl"))

new_popstats = dict((k.name, v) for k, v in pkl["new_popstats"].items())

import matplotlib.pyplot as plt

statlabels = dict(
    a_mean="mean of recurrent term",
    b_mean="mean of input term",
    c_mean="mean of cell state",
    a_var="variance of recurrent term",
    b_var="variance of input term",
    c_var="variance of cell state")

fig, axess = plt.subplots(2, 2, sharex='col')
statss = [["%s_%s" % (key, stat) for key in "ac"]
          for stat in "mean var".split()]
for axes, stats in zip(axess, statss):
    for axis, stat in zip(axes, stats):
        popstat = new_popstats[stat]

        # random subset of popstats
        subset = np.random.choice(popstat.shape[1], size=30, replace=False)

        axis.plot(popstat[:, subset], color="k")
        axis.set_title(statlabels[stat])

# set xlabel only on bottom subplots since x axis is shared
for axis in axess[1]:
    axis.set_xlabel("time steps")

plt.show()

