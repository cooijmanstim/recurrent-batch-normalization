import sys, numpy as np
import cPickle as pkl
import matplotlib.pyplot as plt
from matplotlib import cm

paths = """
LSTM:preliminary/baseline/stats_dimworda_[240]_datamode_top4_usedqsim_1_useelug_0_validFre_1000_clip-c_[10.0]_usebidir_0_encoderq_lstm_dimproj_[240]_use-drop_[True]_optimize_adam_decay-c_[0.0]_truncate_-1_learnh0_1_default.npz.pkl
BN-LSTM:preliminary/batchnorm/stats_dimworda_[240]_datamode_top4_usedqsim_1_useelug_0_validFre_1000_clip-c_[10.0]_usebidir_0_encoderq_bnlstm_dimproj_[240]_use-drop_[True]_optimize_adam_decay-c_[0.0]_truncate_-1_default.npz.pkl
BN-everywhere:preliminary/batchnorm-everywhere/stats_dimworda_[240]_datamode_top4_usedqsim_1_useelug_0_validFre_1000_clip-c_[10.0]_usebidir_0_encoderq_bnlstm_dimproj_[240]_use-drop_[True]_optimize_adam_decay-c_[0.0]_default.npz.pkl
Unidir:reprod/improved_240_uni_bs40_lr8e-4_use_dq_sims1_use_desc_skip_c_g1_sequensewisenorm/stats_bn-bidir-dropout.npz.pkl
Bidir:reprod/bidir/240_bi_bs64_lr8e-5_use_dq_sims1_use_desc_skip_c_g1_sequensewisenorm/stats_bn-bidir-dropout.npz.pkl
""".split()

friendly_labels = {
  "Unidir": "BN-e*",
  "Bidir": "BN-e**",
}

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
        minimum = min(instance["data"][channel_name])
        print label, "min", minimum
        plt.plot(np.asarray(instance["data"][channel_name]), label=label, color=color, linewidth=2, **kwargs)
        if "valid" in channel_name:
          plt.axhline(y=minimum, xmin=0.98, xmax=1, linewidth=2, color=color)
plt.legend(prop=dict(size=12))
plt.xlim((0, 800))
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
  plt.legend(prop=dict(size=12))
  plt.xlim((0, 800))
  plt.ylabel("cost")
  plt.xlabel("training steps")

plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(800 / fig.dpi, 600 / fig.dpi)
plt.savefig("attr_valid2.pdf", bbox_inches="tight")
#plt.show()
