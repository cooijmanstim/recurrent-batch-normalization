import sys
import theano, itertools, pprint, copy, numpy as np, theano.tensor as T, re
from collections import OrderedDict
from blocks.serialization import load
import util

from sequential_mnist import get_stream

# argument: path to a checkpoint file
main_loop = load(sys.argv[1])
print main_loop.log.current_row


# extract population statistic updates
updates = [update for update in main_loop.algorithm.updates
           # FRAGILE
           if re.search("_(mean|var)$", update[0].name)]
print updates

old_popstats = dict((popstat, popstat.get_value()) for popstat, _ in updates)


# baseline doesn't need all this
if updates:
    which_set = "train"
    batch_size = 5000    # -_-
    #nbatches = len(list(main_loop.data_stream.get_epoch_iterator()))
    nbatches = len(list(get_stream(which_set=which_set, batch_size=batch_size).get_epoch_iterator()))

    # destructure moving average expression to construct a new expression
    new_updates = []
    batchstat_name = []
    batchstat_list = []
    for popstat, value in updates:

        print popstat
        batchstat = popstat.tag.estimand
        batchstat_name.append(popstat.name)
        batchstat_list.append(batchstat)

        old_popstats[popstat] = popstat.get_value()

        # FRAGILE: assume population statistics not used in computation of batch statistics
        # otherwise popstat should always have a reasonable value
        popstat.set_value(0 * popstat.get_value(borrow=True))
        new_updates.append((popstat, popstat + batchstat / float(nbatches)))
        #new_updates.append((popstat, batchstat))

    # FRAGILE: assume all the other algorithm updates are unneeded for computation of batch statistics
    estimate_fn = theano.function(main_loop.algorithm.inputs, batchstat_list,
                                  updates=new_updates, on_unused_input="warn")

    bstats = OrderedDict()
    bstats_mean = OrderedDict()
    for n in batchstat_name:
        bstats[n] = []
        bstats_mean[n] = 0.0
    for batch in get_stream(which_set=which_set, batch_size=batch_size).get_epoch_iterator(as_dict=True):
        cur_bstat = estimate_fn(**batch)
        for i in xrange(len(cur_bstat)):
            bstats[batchstat_name[i]].append(cur_bstat[i])
            bstats_mean[batchstat_name[i]] += cur_bstat[i]
    for k, v in bstats_mean.items():
        bstats_mean[k] = v / float(nbatches)
    #for popstat, value in updates:
        #popstat.set_value(bstats_mean[popstat.name])


new_popstats = dict((popstat, popstat.get_value()) for popstat, _ in updates)


from blocks.monitoring.evaluators import DatasetEvaluator
results = dict()
for situation in "training inference".split():
    results[situation] = dict()
    outputs, = [
        extension._evaluator.theano_variables
        for extension in main_loop.extensions
        if getattr(extension, "prefix", None) == "valid_%s" % situation]
    evaluator = DatasetEvaluator(outputs)
    for which_set in "train valid test".split():
        if which_set == "test":
            results[situation][which_set] = evaluator.evaluate(get_stream(which_set=which_set,
                                                                          batch_size=5000))
        else:
            results[situation][which_set] = evaluator.evaluate(get_stream(which_set=which_set,
                                                                          batch_size=5000))

results["proper_test"] = evaluator.evaluate(
    get_stream(
        which_set="test",
        batch_size=1000))
print 'Results: ', results["proper_test"]
import cPickle
cPickle.dump(dict(results=results,
                  old_popstats=old_popstats,
                  new_popstats=new_popstats),
             open(sys.argv[1] + "_popstat_results.pkl", "w"))
