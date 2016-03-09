import sys, os, util
import logging
from collections import OrderedDict
import numpy as np
import theano, theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import blocks.config
import fuel.datasets, fuel.streams, fuel.transformers, fuel.schemes

# i shake my head
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

def softmax_lastaxis(x):
    # for sequence of distributions
    return T.nnet.softmax(x.reshape((-1, x.shape[-1]))).reshape(x.shape)

def crossentropy_lastaxes(yhat, y):
    # for sequence of distributions/targets
    return -(y * T.log(yhat)).sum(axis=yhat.ndim - 1)

class PTB(fuel.datasets.Dataset):
    provides_sources = ('features',)
    example_iteration_scheme = None

    def __init__(self, which_set, length, overlapping=False):
        self.length = length
        self.overlapping = overlapping
        path = os.environ["CHAR_LEVEL_PENNTREE_NPZ"]
        data = np.load(path)
        self.vocab = data["vocab"]
        self.data = data[which_set]
        if self.overlapping:
            self.num_examples = len(self.data) - self.length + 1
        else:
            # drops last ragged batch
            self.num_examples = int(len(self.data) / self.length)
        super(PTB, self).__init__()

    def get_data(self, state=None, request=None):
        if isinstance(request, slice):
            request = list(range(request.start, request.stop, request.step))
        batch = np.zeros((len(request), self.length, len(self.vocab)), dtype=theano.config.floatX)
        for i, start in enumerate(request):
            offset = start
            if not self.overlapping:
                offset *= self.length
            # one-hot
            batch[i, list(range(self.length)), self.data[offset:offset + self.length]] = 1.
        if False:
            #import pdb; pdb.set_trace()
            assert np.allclose(batch.sum(axis=2), 1)
        return (batch,)

def get_stream(which_set, batch_size, length, num_examples=None, overlapping=False):
    dataset = PTB(which_set, length=length, overlapping=overlapping)
    if num_examples is None or num_examples > dataset.num_examples:
        num_examples = dataset.num_examples
    stream = fuel.streams.DataStream.default_stream(
        dataset,
        iteration_scheme=fuel.schemes.ShuffledScheme(num_examples, batch_size))
    return stream

activations = dict(
    tanh=T.tanh,
    identity=lambda x: x,
    relu=lambda x: T.max(0, x))

def bn(x, gammas, betas, args, mean=None, var=None):
    if args.baseline:
        return x + betas, mean, var
    assert x.ndim == 2
    if not args.use_population_statistics:
        mean = x.mean(axis=0)
        var = x.var(axis=0)
    assert mean.ndim == 1
    assert var.ndim == 1
    #var = T.maximum(var, args.epsilon)
    var = var + args.epsilon
    y = theano.tensor.nnet.bn.batch_normalization(
        inputs=x,
        gamma=gammas, beta=betas,
        mean=T.shape_padleft(mean),
        std=T.shape_padleft(T.sqrt(var)))
    assert mean.ndim == 1
    assert var.ndim == 1
    return y, mean, var

class Pain(object):
    pass

class LSTM(object):
    def __init__(self, args, nclasses):
        self.nclasses = nclasses
        self.activation = activations[args.activation]

    def allocate_parameters(self, args):
        if not hasattr(self, "parameters"):
            # dicts suck
            self.parameters = Pain()
            if args.initialization == "identity":
                Wa = np.concatenate([
                    np.eye(args.num_hidden),
                    orthogonal((args.num_hidden, 3 * args.num_hidden)),
                ], axis=1)
            elif args.initialization == "orthogonal":
                Wa = orthogonal((args.num_hidden, 4 * args.num_hidden))
            for parameter in [
                    theano.shared(zeros((args.num_hidden,)), name="h0"),
                    theano.shared(zeros((args.num_hidden,)), name="c0"),
                    theano.shared(Wa.astype(theano.config.floatX), name="Wa"),
                    theano.shared(orthogonal((self.nclasses, 4 * args.num_hidden)), name="Wx"),
                    theano.shared(args.initial_gamma * ones((4 * args.num_hidden,)), name="a_gammas"),
                    theano.shared(args.initial_gamma * ones((4 * args.num_hidden,)), name="b_gammas"),
                    theano.shared(args.initial_beta  * ones((4 * args.num_hidden,)), name="ab_betas"),
                    theano.shared(args.initial_gamma * ones((args.num_hidden,)), name="c_gammas"),
                    theano.shared(args.initial_beta  * ones((args.num_hidden,)), name="c_betas")]:
                add_role(parameter, PARAMETER)
                setattr(self.parameters, parameter.name, parameter)

        # forget gate bias initialization
        ab_betas = self.parameters.ab_betas
        pffft = ab_betas.get_value()
        pffft[args.num_hidden:2*args.num_hidden] = 1.
        ab_betas.set_value(pffft)

        return self.parameters

    def construct_graph(self, args, x, length, popstats=None):
        p = self.allocate_parameters(args)

        # use `symlength` where we need to be able to adapt to longer sequences
        # than the ones we trained on
        symlength = x.shape[0]
        t = T.cast(T.arange(symlength), "int16")
        long_sequence_is_long = T.ge(T.cast(T.arange(symlength), theano.config.floatX), length)
        batch_size = x.shape[1]
        dummy_states = dict(h=T.zeros((symlength, batch_size, args.num_hidden)),
                            c=T.zeros((symlength, batch_size, args.num_hidden)))

        def stepfn(t, long_sequence_is_long, x, dummy_h, dummy_c, h, c, **popstats):
            popstats_by_key = dict()
            for key in "abc":
                popstats_by_key[key] = dict()
                for stat in "mean var".split():
                    if not args.baseline and args.use_population_statistics:
                        popstat = popstats["%s_%s" % (key, stat)]
                        # pluck the appropriate population statistic for this
                        # time step out of the sequence, or take the last
                        # element if we've gone beyond the training length.
                        # if `long_sequence_is_long` then `t` may be unreliable
                        # as it will overflow for looong sequences.
                        popstat = theano.ifelse.ifelse(
                            long_sequence_is_long, popstat[-1], popstat[t])
                    else:
                        popstat = None
                    popstats_by_key[key][stat] = popstat

            atilde, btilde = T.dot(h, p.Wa), T.dot(x, p.Wx)
            a_normal, a_mean, a_var = bn(atilde, p.a_gammas, 0, args, **popstats_by_key["a"])
            b_normal, b_mean, b_var = bn(btilde, p.b_gammas, 0, args, **popstats_by_key["b"])
            ab = a_normal + b_normal + p.ab_betas
            g, f, i, o = [fn(ab[:, j * args.num_hidden:(j + 1) * args.num_hidden])
                          for j, fn in enumerate([self.activation] + 3 * [T.nnet.sigmoid])]
            c = dummy_c + f * c + i * g
            c_normal, c_mean, c_var = bn(c, p.c_gammas, p.c_betas, args, **popstats_by_key["c"])
            h = dummy_h + o * self.activation(c_normal)
            return locals()

        sequences = dict(t=t, x=x, long_sequence_is_long=long_sequence_is_long,
                         dummy_h=dummy_states["h"],
                         dummy_c=dummy_states["c"])
        outputs_info = dict(h=T.repeat(p.h0[None, :], batch_size, axis=0),
                            c=T.repeat(p.c0[None, :], batch_size, axis=0),
                            atilde=None, btilde=None)
        non_sequences = dict()

        if not args.baseline:
            for key, size in zip("abc", [4*args.num_hidden, 4*args.num_hidden, args.num_hidden]):
                for stat, init in zip("mean var".split(), [0, 1]):
                    name = "%s_%s" % (key, stat)

                    if args.use_population_statistics:
                        # population statistics is a sequence, but we pass it in
                        # as a non-sequence and index it ourselves. this allows us
                        # to generalize to longer sequences, in which case we
                        # repeat the last element.
                        non_sequences[name] = popstats[name]
                    else:
                        # provide batch statistic as an output so that we
                        # can estimate population statistics from them.
                        outputs_info[name] = None

        outputs, updates = util.scan(
            stepfn,
            sequences=sequences,
            outputs_info=outputs_info,
            non_sequences=non_sequences)

        if not args.baseline:
            if not args.use_population_statistics:
                # prepare population statistic estimation
                popstats = dict()
                alpha = 0.005
                for key, size in zip("abc", [4*args.num_hidden, 4*args.num_hidden, args.num_hidden]):
                    for stat, init in zip("mean var".split(), [0, 1]):
                        name = "%s_%s" % (key, stat)
                        popstats[name] = theano.shared(
                            init + np.zeros((length, size,),
                                            dtype=theano.config.floatX),
                            name=name)
                        popstats[name].tag.estimand = outputs[name]
                        updates[popstats[name]] = (alpha * outputs[name] +
                                                   (1 - alpha) * popstats[name])

        return outputs, updates, dummy_states, popstats

def construct_common_graph(situation, args, outputs, dummy_states, Wy, by, y):
    ytilde = T.dot(outputs["h"], Wy) + by
    yhat = softmax_lastaxis(ytilde)

    errors = T.neq(T.argmax(y, axis=y.ndim - 1),
                   T.argmax(yhat, axis=yhat.ndim - 1))
    cross_entropies = crossentropy_lastaxes(yhat, y)

    error_rate = errors.mean().copy(name="error_rate")
    cross_entropy = cross_entropies.mean().copy(name="cross_entropy")
    cost = cross_entropy.copy(name="cost")

    graph = ComputationGraph([cost, cross_entropy, error_rate])

    state_grads = dict((k, T.grad(cost, v))
                       for k, v in dummy_states.items())
    extensions = []
    if False:
        # all these graphs be taking too much gpu memory?
        extensions.append(
            DumpVariables("%s_hiddens" % situation, graph.inputs,
                          [v.copy(name="%s%s" % (k, suffix))
                           for suffix, things in [("", outputs), ("_grad", state_grads)]
                           for k, v in things.items()],
                          batch=next(get_stream(which_set="train",
                                                batch_size=args.batch_size,
                                                num_examples=args.batch_size,
                                                length=args.length)
                                     .get_epoch_iterator(as_dict=True)),
                          before_training=True, every_n_epochs=10))

    return graph, extensions

def construct_graphs(args, nclasses):
    constructor = LSTM if args.lstm else RNN

    Wy = theano.shared(orthogonal((args.num_hidden, nclasses)), name="Wy")
    by = theano.shared(np.zeros((nclasses,), dtype=theano.config.floatX), name="by")
    for parameter in [Wy, by]:
        add_role(parameter, PARAMETER)

    x = T.tensor3("features")

    #theano.config.compute_test_value = "warn"
    #x.tag.test_value = np.random.random((7, args.length, nclasses)).astype(theano.config.floatX)

    # move time axis forward
    x = x.dimshuffle(1, 0, 2)
    # task is to predict next character
    x, y = x[:-1], x[1:]
    length = args.length - 1

    args.use_population_statistics = False
    turd = constructor(args, nclasses)
    (outputs, training_updates, dummy_states, popstats) = turd.construct_graph(
        args, x, length)
    training_graph, training_extensions = construct_common_graph("training", args, outputs, dummy_states, Wy, by, y)
    args.use_population_statistics = True
    (outputs, inference_updates, dummy_states, _) = turd.construct_graph(
        args, x, length,
        # use popstats from previous invocation
        popstats=popstats)
    inference_graph, inference_extensions = construct_common_graph("inference", args, outputs, dummy_states, Wy, by, y)
    args.use_population_statistics = False

    # pfft
    return (dict(training=training_graph,      inference=inference_graph),
            dict(training=training_extensions, inference=inference_extensions),
            dict(training=training_updates,    inference=inference_updates))

if __name__ == "__main__":
    nclasses = 50

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--num-hidden", type=int, default=1000)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--lstm", action="store_true")
    parser.add_argument("--initialization", choices="identity orthogonal".split(), default="identity")
    parser.add_argument("--initial-gamma", type=float, default=1e-1)
    parser.add_argument("--initial-beta", type=float, default=0)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--activation", choices=list(activations.keys()), default="tanh")
    parser.add_argument("--optimizer", choices="sgdmomentum rmsprop", default="rmsprop")
    parser.add_argument("--continue-from")
    args = parser.parse_args()

    np.random.seed(args.seed)
    blocks.config.config.default_seed = args.seed

    if args.continue_from:
        from blocks.serialization import load
        main_loop = load(args.continue_from)
        main_loop.run()
        sys.exit(0)

    graphs, extensions, updates = construct_graphs(args, nclasses)

    ### optimization algorithm definition
    if args.optimizer == "rmsprop":
        optimizer = RMSProp(learning_rate=args.learning_rate, decay_rate=0.9)
    elif args.optimizer == "sgdmomentum":
        optimizer = Momentum(learning_rate=args.learning_rate, momentum=0.99)
    step_rule = CompositeRule([
        StepClipping(1.),
        optimizer,
    ])
    algorithm = GradientDescent(cost=graphs["training"].outputs[0],
                                parameters=graphs["training"].parameters,
                                step_rule=step_rule)
    algorithm.add_updates(updates["training"])
    model = Model(graphs["training"].outputs[0])
    extensions = extensions["training"] + extensions["inference"]

    # step monitor
    step_channels = []
    step_channels.extend([
        algorithm.steps[param].norm(2).copy(name="step_norm:%s" % name)
        for name, param in model.get_parameter_dict().items()])
    step_channels.append(algorithm.total_step_norm.copy(name="total_step_norm"))
    step_channels.append(algorithm.total_gradient_norm.copy(name="total_gradient_norm"))
    step_channels.extend(graphs["training"].outputs)
    logger.warning("constructing training data monitor")
    extensions.append(TrainingDataMonitoring(
        step_channels, prefix="iteration", after_batch=True))

    # parameter monitor
    extensions.append(DataStreamMonitoring(
        [param.norm(2).copy(name="parameter.norm:%s" % name)
         for name, param in model.get_parameter_dict().items()],
        data_stream=None, after_epoch=True))

    # performance monitor
    for situation in "training".split(): #"training inference".split():
        for which_set in "train valid test".split():
            logger.warning("constructing %s %s monitor" % (which_set, situation))
            channels = list(graphs[situation].outputs)
            extensions.append(DataStreamMonitoring(
                channels,
                prefix="%s_%s" % (which_set, situation), after_epoch=True,
                data_stream=get_stream(which_set=which_set, batch_size=args.batch_size,
                                       num_examples=50000, length=args.length)))

    extensions.extend([
        TrackTheBest("valid_training_error_rate", "best_valid_training_error_rate"),
        DumpBest("best_valid_training_error_rate", "best.zip"),
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
        data_stream=get_stream(which_set="train", batch_size=args.batch_size, length=args.length),
        algorithm=algorithm, extensions=extensions, model=model)
    main_loop.run()
