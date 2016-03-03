import sys
import theano, itertools, pprint, copy, numpy as np, theano.tensor as T, re
from collections import OrderedDict
from blocks.serialization import load
import util

from penntreebank import PTB, get_stream

main_loop = load(sys.argv[1])

# extract population statistic updates
updates = [update for update in main_loop.algorithm.updates
           # FRAGILE
           if re.search("_(mean|var)$", update[0].name)]
print updates

# -_-
nbatches = len(list(main_loop.data_stream.get_epoch_iterator()))

old_popstats = dict((popstat, popstat.get_value()) for popstat, _ in updates)

# destructure moving average expression to construct a new expression
new_updates = []
for popstat, value in updates:
    # FRAGILE
    assert value.owner.op.scalar_op == theano.scalar.add
    terms = value.owner.inputs
    # right multiplicand is hostfromgpu(popstat)
    assert terms[1].owner.inputs[1].owner.inputs[0] == popstat
    batchstat = terms[0].owner.inputs[1]

    old_popstats[popstat] = popstat.get_value()

    # FRAGILE: assume population statistics not used in computation of batch statistics
    # otherwise popstat should always have a reasonable value
    popstat.set_value(0 * popstat.get_value(borrow=True))
    new_updates.append((popstat, popstat + batchstat / float(nbatches)))

# FRAGILE: assume all the other algorithm updates are unneeded for computation of batch statistics
estimate_fn = theano.function(main_loop.algorithm.inputs, [],
                              updates=new_updates, on_unused_input="warn")
for batch in main_loop.data_stream.get_epoch_iterator(as_dict=True):
    estimate_fn(**batch)

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
        results[situation][which_set] = OrderedDict(
            (length, evaluator.evaluate(get_stream(
                which_set=which_set,
                batch_size=1000,
                length=length)))
            for length in [50, 100, 200, 300, 400, 500])

import cPickle
cPickle.dump(dict(results=results,
                  old_popstats=old_popstats,
                  new_popstats=new_popstats),
             open(sys.argv[1] + "_popstat_results.pkl", "w"))
