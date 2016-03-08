import sys
import logging
from collections import OrderedDict
import numpy as np
import theano, theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import blocks.config
import fuel.datasets, fuel.streams, fuel.transformers, fuel.schemes

### optimization algorithm definition
from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent, RMSProp, StepClipping, CompositeRule, Momentum
from blocks.model import Model
from blocks.extensions import FinishAfter, Printing, ProgressBar, Timing
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.training import TrackTheBest
from blocks.extensions.saveload import Checkpoint
from extensions import DumpLog, DumpBest, PrintingTo, DumpVariables
from blocks.main_loop import MainLoop
from blocks.utils import shared_floatx_zeros
from blocks.roles import add_role, PARAMETER

import util


logging.basicConfig()
logger = logging.getLogger(__name__)

def zeros(shape):
    return np.zeros(shape, dtype=theano.config.floatX)

def ones(shape):
    return np.ones(shape, dtype=theano.config.floatX)

def glorot(shape):
    d = np.sqrt(6. / sum(shape))
    return np.random.uniform(-d, +d, size=shape).astype(theano.config.floatX)

def orthogonal(shape):
    # taken from https://gist.github.com/kastnerkyle/f7464d98fe8ca14f2a1a
    """ benanne lasagne ortho init (faster than qr approach)"""
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v  # pick the one with the correct shape
    q = q.reshape(shape)
    return q[:shape[0], :shape[1]].astype(theano.config.floatX)

_datasets = None
def get_dataset(which_set):
    global _datasets
    if not _datasets:
        MNIST = fuel.datasets.MNIST
        # jump through hoops to instantiate only once and only if needed
        _datasets = dict(
            train=MNIST(which_sets=["train"], subset=slice(None, 50000)),
            valid=MNIST(which_sets=["train"], subset=slice(50000, None)),
            test=MNIST(which_sets=["test"]))
    return _datasets[which_set]

def get_stream(which_set, batch_size, num_examples=None):
    dataset = get_dataset(which_set)
    if num_examples is None or num_examples > dataset.num_examples:
        num_examples = dataset.num_examples
    stream = fuel.streams.DataStream.default_stream(
        dataset,
        iteration_scheme=fuel.schemes.ShuffledScheme(num_examples, batch_size))
    return stream


# def construct_rnn(args, x, activation):
#     parameters = []

#     h0 = theano.shared(zeros((args.num_hidden,)), name="h0")
#     Wh = theano.shared((0.99 if args.baseline else 1) * np.eye(args.num_hidden, dtype=theano.config.floatX), name="Wh")
#     Wx = theano.shared(orthogonal((1, args.num_hidden)), name="Wx")

#     parameters.extend([h0, Wh, Wx])

#     gammas = theano.shared(args.initial_gamma * ones((args.num_hidden,)), name="gammas")
#     betas  = theano.shared(args.initial_beta  * ones((args.num_hidden,)), name="betas")

#     if args.baseline:
#         parameters.extend([betas])
#         def bn(x, gammas, betas):
#             return x + betas
#     else:
#         parameters.extend([gammas, betas])
#         def bn(x, gammas, betas):
#             mean, var = x.mean(axis=0, keepdims=True), x.var(axis=0, keepdims=True)
#             # if only
#             mean.tag.batchstat, var.tag.batchstat = True, True
#             #var = T.maximum(var, args.epsilon)
#             var = var + args.epsilon
#             return (x - mean) / T.sqrt(var) * gammas + betas

#     xtilde = T.dot(x, Wx)

#     if args.noise:
#         # prime h with white noise
#         Trng = MRG_RandomStreams()
#         h_prime = Trng.normal((xtilde.shape[1], args.num_hidden), std=args.noise)
#     elif args.summarize:
#         # prime h with mean of example
#         h_prime = x.mean(axis=[0, 2])[:, None]
#     else:
#         h_prime = 0

#     dummy_states = dict(h     =T.zeros((xtilde.shape[0], xtilde.shape[1], args.num_hidden)),
#                         htilde=T.zeros((xtilde.shape[0], xtilde.shape[1], args.num_hidden)))

#     def stepfn(xtilde, dummy_h, dummy_htilde, h):
#         htilde = dummy_htilde + T.dot(h, Wh) + xtilde
#         h = dummy_h + activation(bn(htilde, gammas, betas))
#         return h, htilde

#     [h, htilde], _ = theano.scan(stepfn,
#                                  sequences=[xtilde, dummy_states["h"], dummy_states["htilde"]],
#                                  outputs_info=[T.repeat(h0[None, :], xtilde.shape[1], axis=0) + h_prime,
#                                                None])

#     return dict(h=h, htilde=htilde), dummy_states, parameters

# def construct_lstm(args, x, activation):
#     parameters = []

#     h0 = theano.shared(zeros((args.num_hidden,)), name="h0")
#     c0 = theano.shared(zeros((args.num_hidden,)), name="c0")
#     Wa = theano.shared(np.concatenate([
#         np.eye(args.num_hidden),
#         orthogonal((args.num_hidden, 3 * args.num_hidden)),
#     ], axis=1).astype(theano.config.floatX), name="Wa")

#     #Wa = theano.shared(np.concatenate([
#     #    orthogonal((args.num_hidden, args.num_hidden)),
#     #    orthogonal((args.num_hidden, 3 * args.num_hidden)),
#     #], axis=1).astype(theano.config.floatX), name="Wa")
#     Wx = theano.shared(orthogonal((1, 4 * args.num_hidden)), name="Wx")

#     parameters.extend([h0, c0, Wa, Wx])

#     a_gammas = theano.shared(args.initial_gamma * ones((4 * args.num_hidden,)), name="a_gammas")
#     b_gammas = theano.shared(args.initial_gamma * ones((4 * args.num_hidden,)), name="b_gammas")
#     ab_betas = theano.shared(args.initial_beta  * ones((4 * args.num_hidden,)), name="ab_betas")
#     h_gammas = theano.shared(args.initial_gamma * ones((args.num_hidden,)), name="h_gammas")
#     h_betas  = theano.shared(args.initial_beta  * ones((args.num_hidden,)), name="h_betas")

#     # forget gate bias initialization
#     pffft = ab_betas.get_value()
#     pffft[args.num_hidden:2*args.num_hidden] = 1.
#     ab_betas.set_value(pffft)

#     if args.baseline:
#         parameters.extend([ab_betas, h_betas])
#         def bn(x, gammas, betas):
#             return x + betas
#     else:
#         parameters.extend([a_gammas, b_gammas, h_gammas, ab_betas, h_betas])
#         def bn(x, gammas, betas):
#             mean, var = x.mean(axis=0, keepdims=True), x.var(axis=0, keepdims=True)
#             # if only
#             mean.tag.batchstat, var.tag.batchstat = True, True
#             #var = T.maximum(var, args.epsilon)
#             var = var + args.epsilon
#             return (x - mean) / T.sqrt(var) * gammas + betas

#     xtilde = T.dot(x, Wx)

#     if args.noise:
#         # prime h with white noise
#         Trng = MRG_RandomStreams()
#         h_prime = Trng.normal((xtilde.shape[1], args.num_hidden), std=args.noise)
#     elif args.summarize:
#         # prime h with mean of example
#         h_prime = x.mean(axis=[0, 2])[:, None]
#     else:
#         h_prime = 0

#     dummy_states = dict(h=T.zeros((xtilde.shape[0], xtilde.shape[1], args.num_hidden)),
#                         c=T.zeros((xtilde.shape[0], xtilde.shape[1], args.num_hidden)))

#     def stepfn(xtilde, dummy_h, dummy_c, h, c):
#         atilde = T.dot(h, Wa)
#         btilde = xtilde
#         a = bn(atilde, a_gammas, ab_betas)
#         b = bn(btilde, b_gammas, 0)
#         #b = btilde
#         ab = a + b
#         g, f, i, o = [fn(ab[:, j * args.num_hidden:(j + 1) * args.num_hidden])
#                       for j, fn in enumerate([activation] + 3 * [T.nnet.sigmoid])]
#         c = dummy_c + f * c + i * g
#         htilde = c
#         h = dummy_h + o * activation(bn(htilde, h_gammas, h_betas))
#         return h, c, atilde, btilde, htilde

#     [h, c, atilde, btilde, htilde], _ = theano.scan(
#         stepfn,
#         sequences=[xtilde, dummy_states["h"], dummy_states["c"]],
#         outputs_info=[T.repeat(h0[None, :], xtilde.shape[1], axis=0) + h_prime,
#                       T.repeat(c0[None, :], xtilde.shape[1], axis=0),
#                       None, None, None])
#     return dict(h=h, c=c, atilde=atilde, btilde=btilde, htilde=htilde), dummy_states, parameters

def bn(x, gammas, betas, mean, var, args):
    assert mean.ndim == 1
    assert var.ndim == 1
    assert x.ndim == 2
    if not args.use_population_statistics:
        mean = x.mean(axis=0)
        var = x.var(axis=0)
    #var = T.maximum(var, args.epsilon)
    var = var + args.epsilon

    if args.baseline:
        y = x + betas
    else:
        y = theano.tensor.nnet.bn.batch_normalization(
            inputs=x, gamma=gammas, beta=betas,
            mean=T.shape_padleft(mean), std=T.shape_padleft(T.sqrt(var)),
            mode="high_mem")
    assert mean.ndim == 1
    assert var.ndim == 1
    return y, mean, var

activations = dict(
    tanh=T.tanh,
    identity=lambda x: x,
    relu=lambda x: T.max(0, x))


class Empty(object):
    pass

class LSTM(object):
    def __init__(self, args, nclasses):
        self.nclasses = nclasses
        self.activation = activations[args.activation]

    def allocate_parameters(self, args):
        if not hasattr(self, "parameters"):
            self.parameters = Empty()
            parameters_list = [
                theano.shared(zeros((args.num_hidden,)), name="h0"),
                theano.shared(zeros((args.num_hidden,)), name="c0"),
                theano.shared(np.concatenate([np.eye(args.num_hidden),
                                              orthogonal((args.num_hidden, 3 * args.num_hidden)),],
                                             axis=1).astype(theano.config.floatX), name="Wa"),
                theano.shared(orthogonal((1, 4 * args.num_hidden)), name="Wx"),
                theano.shared(args.initial_gamma * ones((4 * args.num_hidden,)), name="a_gammas"),
                theano.shared(args.initial_gamma * ones((4 * args.num_hidden,)), name="b_gammas"),
                theano.shared(args.initial_beta  * ones((4 * args.num_hidden,)), name="ab_betas"),
                theano.shared(args.initial_gamma * ones((args.num_hidden,)), name="c_gammas"),
                theano.shared(args.initial_beta  * ones((args.num_hidden,)), name="c_betas")]

            for parameter in parameters_list:
                add_role(parameter, PARAMETER)
                setattr(self.parameters, parameter.name, parameter)

        # forget gate bias initialization
        ab_betas = self.parameters.ab_betas
        forget_biais = ab_betas.get_value()
        forget_biais[args.num_hidden:2*args.num_hidden] = 1.
        ab_betas.set_value(forget_biais)
        return self.parameters

    def construct_graph(self, args, x, length, popstats=None):
        p = self.allocate_parameters(args)

        def stepfn(x, dummy_h, dummy_c, h, c,
                   a_mean, b_mean, c_mean,
                   a_var, b_var, c_var):

            atilde = T.dot(h, p.Wa)
            btilde = x
            a_normal, a_mean, a_var = bn(atilde, p.a_gammas, p.ab_betas, a_mean, a_var, args)
            b_normal, b_mean, b_var = bn(btilde, p.b_gammas, 0,          b_mean, b_var, args)
            ab = a_normal + b_normal
            g, f, i, o = [fn(ab[:, j * args.num_hidden:(j + 1) * args.num_hidden])
                          for j, fn in enumerate([self.activation] + 3 * [T.nnet.sigmoid])]
            c = dummy_c + f * c + i * g
            c_normal, c_mean, c_var = bn(c, p.c_gammas, p.c_betas, c_mean, c_var, args)
            h = dummy_h + o * self.activation(c_normal)
            return locals()


        symlength = x.shape[0]
        batch_size = x.shape[1]
        dummy_states = dict(h=T.zeros((symlength, batch_size, args.num_hidden)),
                            c=T.zeros((symlength, batch_size, args.num_hidden)))

        xtilde = T.dot(x, p.Wx)
        sequences = dict(x=xtilde,
                         dummy_h=dummy_states["h"],
                         dummy_c=dummy_states["c"])
        outputs_info = dict(h=T.repeat(p.h0[None, :], batch_size, axis=0),
                            c=T.repeat(p.c0[None, :], batch_size, axis=0),
                            atilde=None, btilde=None)
        non_sequences = dict()

        for key, size in zip("abc", [4*args.num_hidden, 4*args.num_hidden, args.num_hidden]):
            for stat, init in zip("mean var".split(), [0, 1]):
                name = "%s_%s" % (key, stat)
                if popstats:
                    ### All sequence are fixed size for sequential mnist
                    sequences[name] =  popstats[name][:symlength]
                else:
                    # provide outputs to allow the stepfn to estimate batch
                    # statistics as recurrent running average
                    outputs_info[name] = init + T.zeros((size,))

        outputs, updates = util.scan(
            stepfn,
            sequences=sequences,
            outputs_info=outputs_info,
            non_sequences=non_sequences)

        if not popstats:
            # prepare population statistic estimation
            popstats = dict()
            alpha = 1e-2
            for key, size in zip("abc", [4*args.num_hidden, 4*args.num_hidden, args.num_hidden]):
                for stat, init in zip("mean var".split(), [0, 1]):
                    name = "%s_%s" % (key, stat)
                    popstats[name] = theano.shared(
                        init + np.zeros((length, size,), dtype=theano.config.floatX),
                        name=name)
                    popstats[name].tag.estimand = outputs[name]
                    ## FIXME we don't need to estimate the popstat here
                    #updates[popstats[name]] = (alpha * outputs[name] +
                    #                           (1 - alpha) * popstats[name])

            return outputs, updates, dummy_states, popstats


def construct_common_graph(situation, args, outputs, dummy_states, Wy, by, y):
    ytilde = T.dot(outputs["h"][-1], Wy) + by
    yhat = T.nnet.softmax(ytilde)

    errors = T.neq(y, T.argmax(yhat, axis=1))
    cross_entropies = T.nnet.categorical_crossentropy(yhat, y)

    error_rate = errors.mean().copy(name="error_rate")
    cross_entropy = cross_entropies.mean().copy(name="cross_entropy")
    cost = cross_entropy.copy(name="cost")
    graph = ComputationGraph([cost, cross_entropy, error_rate])

    state_grads = dict((k, T.grad(cost, v)) for k, v in dummy_states.items())

    extensions = []
    # extensions = [
    #     DumpVariables("%s_hiddens" % situation, graph.inputs,
    #                   [v.copy(name="%s%s" % (k, suffix))
    #                    for suffix, things in [("", outputs), ("_grad", state_grads)]
    #                    for k, v in things.items()],
    #                   batch=next(get_stream(which_set="train",
    #                                         batch_size=args.batch_size,
    #                                         num_examples=args.batch_size)
    #                              .get_epoch_iterator(as_dict=True)),
    #                   before_training=True, every_n_epochs=10)]

    return graph, extensions

def construct_graphs(args, nclasses, length):
    constructor = LSTM if args.lstm else RNN

    Wy = theano.shared(orthogonal((args.num_hidden, nclasses)), name="Wy")
    by = theano.shared(np.zeros((nclasses,), dtype=theano.config.floatX), name="by")
    add_role(Wy, PARAMETER)
    add_role(by, PARAMETER)


    ### graph construction
    inputs = dict(features=T.tensor4("features"), targets=T.imatrix("targets"))
    x, y = inputs["features"], inputs["targets"]

    #theano.config.compute_test_value = "warn"
    #x.tag.test_value = np.random.random((args.batch_size, 28, 28, 1)).astype(theano.config.floatX)
    #y.tag.test_value = np.ones((args.batch_size, 1)).astype(np.int32)

    x = x.reshape((x.shape[0], length, 1))
    y = y.flatten(ndim=1)
    x = x.dimshuffle(1, 0, 2)

    if args.permuted:
        permutation = np.random.randint(0, length, size=(length,))
        x = x[permutation]

    args.use_population_statistics = False
    turd = constructor(args, nclasses)
    (outputs, training_updates, dummy_states, popstats) = turd.construct_graph(
        args, x, length)
    training_graph, training_extensions = construct_common_graph("training", args, outputs, dummy_states, Wy, by, y)
    args.use_population_statistics = True
    (outputs, inference_updates, dummy_states, _) = turd.construct_graph(
        args, x, length)
     # use popstats from previous invocation
     #popstats=popstats)
    inference_graph, inference_extensions = construct_common_graph("inference", args, outputs, dummy_states, Wy, by, y)
    args.use_population_statistics = False
    return (dict(training=training_graph,      inference=inference_graph),
            dict(training=training_extensions, inference=inference_extensions),
            dict(training=training_updates,    inference=inference_updates))

if __name__ == "__main__":
    sequence_length = 784
    nclasses = 10

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=100)
    #parser.add_argument("--noise", type=float, default=None)
    #parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--num-hidden", type=int, default=100)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--lstm", action="store_true")
    parser.add_argument("--initial-gamma", type=float, default=0.1)
    parser.add_argument("--initial-beta", type=float, default=0)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--activation", choices=list(activations.keys()), default="tanh")
    parser.add_argument("--continue-from")
    parser.add_argument("--permuted", action="store_true")
    args = parser.parse_args()

    #assert not (args.noise and args.summarize)
    np.random.seed(args.seed)
    blocks.config.config.default_seed = args.seed

    if args.continue_from:
        from blocks.serialization import load
        main_loop = load(args.continue_from)
        main_loop.run()
        sys.exit(0)

    graphs, extensions, updates = construct_graphs(args, nclasses, sequence_length)

    ### optimization algorithm definition
    step_rule = CompositeRule([
        StepClipping(1.),
        #Momentum(learning_rate=args.learning_rate, momentum=0.9),
        RMSProp(learning_rate=args.learning_rate, decay_rate=0.9),
    ])

    import pdb; pdb.set_trace()
    algorithm = GradientDescent(cost=graphs["training"].outputs[0],
                                parameters=graphs["training"].parameters,
                                step_rule=step_rule)
    algorithm.add_updates(updates["training"])
    model = Model(graphs["training"].outputs[0])
    extensions = extensions["training"] #+ extensions["inference"]

    # step monitor (after epoch to limit the log size)
    step_channels = []
    step_channels.extend([
        algorithm.steps[param].norm(2).copy(name="step_norm:%s" % name)
        for name, param in model.get_parameter_dict().items()])
    step_channels.append(algorithm.total_step_norm.copy(name="total_step_norm"))
    step_channels.append(algorithm.total_gradient_norm.copy(name="total_gradient_norm"))
    step_channels.extend(graphs["training"].outputs)
    logger.warning("constructing training data monitor")
    extensions.append(TrainingDataMonitoring(
        step_channels, prefix="iteration", after_batch=False))

    # parameter monitor
    extensions.append(DataStreamMonitoring(
        [param.norm(2).copy(name="parameter.norm:%s" % name)
         for name, param in model.get_parameter_dict().items()],
        data_stream=None, after_epoch=True))

    # performance monitor
    for situation in "training inference".split():
        for which_set in "train valid test".split():
            logger.warning("constructing %s %s monitor" % (which_set, situation))
            channels = list(graphs[situation].outputs)
            extensions.append(DataStreamMonitoring(
                channels,
                prefix="%s_%s" % (which_set, situation), after_epoch=True,
                data_stream=get_stream(which_set=which_set, batch_size=args.batch_size))) #, num_examples=1000)))

    extensions.extend([
        TrackTheBest("valid_training_error_rate", "best_valid_training_error_rate"),
        DumpBest("best_valid_traing_error_rate", "best.zip"),
        FinishAfter(after_n_epochs=args.num_epochs),
        #FinishIfNoImprovementAfter("best_valid_error_rate", epochs=50),
        Checkpoint("checkpoint.zip", on_interrupt=False, every_n_epochs=1, use_cpickle=True),
        DumpLog("log.pkl", after_epoch=True)])

    if not args.cluster:
        extensions.append(ProgressBar())

    extensions.extend([
        Timing(),
        Printing(),
        PrintingTo("log"),
    ])
    main_loop = MainLoop(
        data_stream=get_stream(which_set="train", batch_size=args.batch_size),
        algorithm=algorithm, extensions=extensions, model=model)
    main_loop.run()
