import sys
# python i hate you
sys.path.insert(0, "/u/cooijmat/dev/wiprepos/Theano")
sys.path.append("/u/cooijmat/dev/wiprepos/Theano")
import theano, itertools, pprint, copy, numpy as np, theano.tensor as T
from collections import OrderedDict
from theano.gof.op import ops_with_inner_function
from theano.scan_module.scan_op import Scan
from theano.scan_module.scan_utils import scan_args
from blocks.serialization import load

def equizip(*sequences):
    sequences = list(map(list, sequences))
    assert all(len(sequence) == len(sequences[0]) for sequence in sequences[1:])
    return zip(*sequences)

# get outer versions of the given inner variables of a scan node
def export(node, extra_inner_outputs):
    assert isinstance(node.op, Scan)

    old_inner_inputs = node.op.inputs
    old_inner_outputs = node.op.outputs
    old_outer_inputs = node.inputs

    new_inner_inputs = list(old_inner_inputs)
    new_inner_outputs = list(old_inner_outputs)
    new_outer_inputs = list(old_outer_inputs)
    new_info = copy.deepcopy(node.op.info)

    # put the new inner outputs in the right place in the output list and
    # update info
    new_info["n_nit_sot"] += len(extra_inner_outputs)
    yuck = len(old_inner_outputs) - new_info["n_shared_outs"]
    new_inner_outputs[yuck:yuck] = extra_inner_outputs

    # scan() adds an outer input for each nitsot. we need to do this too.
    # luckily it's the same input for each of them, so we can use a reference
    # to the ones already there.
    # if there was no nitsot in the old op we're shit out of luck
    assert node.op.n_nit_sot
    # logic taken from Scan.outer_nitsot
    offset = (1 + node.op.n_seqs + node.op.n_mit_mot + node.op.n_mit_sot +
              node.op.n_sit_sot + node.op.n_shared_outs)
    # take the first outer nitsot input and repeat it
    new_outer_inputs[offset:offset] = [new_outer_inputs[offset]] * len(extra_inner_outputs)

    new_op = Scan(new_inner_inputs, new_inner_outputs, new_info)
    outer_outputs = new_op(*new_outer_inputs)

    # grab the outputs we actually care about
    extra_outer_outputs = outer_outputs[yuck:yuck + len(extra_inner_outputs)]
    return extra_outer_outputs

def gather_symbatchstats_and_estimators(outputs):
    symbatchstats = []
    estimators = []

    for var in theano.gof.graph.ancestors(outputs):
        if hasattr(var.tag, "batchstat"):
            symbatchstats.append(var)
            estimators.append(var)

        # descend into Scan/OpFromGraph
        try:
            op = var.owner.op
        except:
            continue
        if op.__class__ in ops_with_inner_function:
            print "descending into", var

            inner_estimators, inner_symbatchstats = gather_symbatchstats_and_estimators(op.outputs)
            outer_estimators = export(var.owner, inner_estimators)

            # take mean of each of outer_outputs along axis 0 to
            # average the estimate across the sequence
            outer_estimators = [x.mean(axis=0) for x in outer_estimators]

            symbatchstats.extend(inner_symbatchstats)
            estimators.extend(outer_estimators)

    return symbatchstats, estimators

def get_population_outputs(inputs, batch_outputs, estimation_batches):
    symbatchstats, estimators = gather_symbatchstats_and_estimators(batch_outputs)
    print "symbatchstats x estimators", zip(symbatchstats, estimators)

    assert symbatchstats

    # take average of batch statistics over training set
    estimator_fn = theano.function(inputs, estimators, on_unused_input="warn")
    popstats_by_symbatchstat = {}
    for i, batch in enumerate(estimation_batches):
        estimates = estimator_fn(**batch)
        for symbatchstat, estimator, estimate in equizip(symbatchstats, estimators, estimates):
            if symbatchstat not in popstats_by_symbatchstat:
                popstats_by_symbatchstat[symbatchstat] = np.zeros_like(estimate)
            popstats_by_symbatchstat[symbatchstat] *= i / float(i + 1)
            popstats_by_symbatchstat[symbatchstat] += 1 / float(i + 1) * estimate

    population_replacements = [
        (symbatchstat,
         # need as_tensor_variable to make sure it's not a CudaNdarray
         # because then the replacement will fail as symbatchstat has not
         # been moved to the gpu yet.
         T.as_tensor_variable(
             T.patternbroadcast(theano.shared(popstat),
                                symbatchstat.broadcastable))
         .copy(name="popstat_%s" % symbatchstat.name))
        for symbatchstat, popstat in popstats_by_symbatchstat.items()]
    print "population replacements", population_replacements

    if False:
        # clone doesn't replace inside scan
        from theano.scan_module.scan_utils import clone
        population_outputs = clone(batch_outputs, replace=population_replacements)
    else:
        from theano.scan_module.scan_utils import map_variables
        # work around cloning
        aargh = {}
        for k, v in population_replacements:
            k.tag.original_id = id(k)
            aargh[k.tag.original_id] = v
        population_outputs = map_variables(
            lambda var: (aargh[var.tag.original_id]
                         if hasattr(var.tag, "original_id")
                         else var),
            batch_outputs)

    return population_outputs
