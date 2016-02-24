import sys, os, pprint, math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from blocks.serialization import load

statistics = dict(batchmean=lambda x: x.mean(axis=1),
                  batchvar=lambda x: x.var(axis=1),
                  #minbatchvar=lambda x: x.var(axis=1).min(axis=1),
                  #maxbatchvar=lambda x: x.var(axis=1).max(axis=1),
                  norm=lambda x: np.sqrt((x**2).sum(axis=(1, 2))))

paths = dict(enumerate(sys.argv[1:]))
pprint.pprint(paths)
subplotrows = int(math.ceil(math.sqrt(len(paths))))

instances = dict((k, load(v)) for k, v in paths.items())

keys = list(next(iter(instances.values())).keys())
print keys
for key in keys:
    for statlabel, statistic in statistics.items():
        figure, axes = plt.subplots(subplotrows, subplotrows)
        if subplotrows == 1:
            # morons
            axes = [[axes]]
        label = 0
        for i in range(subplotrows):
            for j in range(subplotrows):
                try:
                    instance = instances[label]
                except:
                    continue
                result = statistic(instance[key])
                axis = axes[i][j]
                if result.ndim == 1:
                    # logarithmic line plot
                    axis.plot(result)
                    axis.set_yscale("log")
                elif result.ndim == 2:
                    # heatmap
                    mappable = axis.imshow(result.T, cmap="bone", interpolation="none", aspect="auto")
                    divider = make_axes_locatable(axis)
                    figure.colorbar(mappable, cax=divider.append_axes("right", size="5%", pad=0.05))
                axis.set_title("#%i" % label)
        title = "%s %s" % (key, statlabel)
        figure.suptitle(title)
        figure.canvas.set_window_title(title)
import pdb; pdb.set_trace()
plt.show()
