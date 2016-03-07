import sys, cPickle
import matplotlib.pyplot as plt

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
    with open(path, "rb") as file:
        thing = cPickle.load(file)
    return dict(name=name, path=path, **thing)

paths = sys.argv[1:]
instances = list(map(load_instance, enumerate(paths)))

import math
def natstobits(x):
    return x / math.log(2)

colors = "blue green red cyan magenta yellow black white".split()
for which_set in "train valid".split():
    plt.figure()
    for situation, kwargs in dict(training=dict(linestyle="dashed"),
                                  inference=dict(linestyle="solid")).items():
        for color, instance in zip(colors, instances):
            label = "%s, %s" % (instance["name"],
                                dict(training="batch",
                                     inference="population")[situation])
            results = instance["results"][situation][which_set]
            tvs = [(t, v["cross_entropy"]) for t, v in results.items()]
            time, value = zip(*tvs)
            value = list(map(natstobits, value))
            plt.plot(time, value, label=label, c=color, **kwargs)
            #plt.yscale("log")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(which_set + " set")
    plt.xlabel("sequence length (trained on 50)")
    plt.ylabel("bits per character")

for instance in instances:
    print "bpc on full test", instance["name"], natstobits(instance["results"]["proper_test"]["cross_entropy"])

import pdb; pdb.set_trace()
plt.show()
import pdb; pdb.set_trace()
