import sys, os, util, functools
import logging
from collections import OrderedDict
import numpy as np
import theano, theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import blocks.config
import fuel.datasets, fuel.streams, fuel.transformers, fuel.schemes

from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent, RMSProp, StepClipping, CompositeRule, Momentum, Adam
from blocks.model import Model
from blocks.extensions import FinishAfter, Printing, ProgressBar, Timing
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks.extensions.stopping import FinishIfNoImprovementAfter
from blocks.extensions.training import TrackTheBest
from blocks.extensions.saveload import Checkpoint
from extensions import DumpLog, DumpBest, PrintingTo, DumpVariables, SharedVariableModifier
from blocks.main_loop import MainLoop
from blocks.utils import shared_floatx_zeros
from blocks.roles import add_role, PARAMETER

logging.basicConfig()
logger = logging.getLogger(__name__)

def learning_rate_decayer(decay_rate, i, learning_rate):
    return ((1. - decay_rate) * learning_rate).astype(theano.config.floatX)

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

def uniform(shape, scale):
    return np.random.uniform(-scale, +scale, size=shape).astype(theano.config.floatX)

def softmax_lastaxis(x):
    # for sequence of distributions
    return T.nnet.softmax(x.reshape((-1, x.shape[-1]))).reshape(x.shape)

def crossentropy_lastaxes(yhat, y):
    # for sequence of distributions/targets
    return -(y * T.log(yhat)).sum(axis=yhat.ndim - 1)

_data_cache = dict()
def get_data(which_set):
    if which_set not in _data_cache:
        path = os.environ["CHAR_LEVEL_PENNTREE_NPZ"]
        data = np.load(path)
        # put the entire thing on GPU in one-hot (takes
        # len(self.vocab) * len(self.data) * sizeof(floatX) bytes
        # which is about 1G for the training set and less for the
        # other sets)
        CudaNdarray = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.CudaNdarray
        # (doing it in numpy first because cudandarray doesn't accept
        # lists of indices)
        one_hot_data = np.eye(len(data["vocab"]), dtype=theano.config.floatX)[data[which_set]]
        _data_cache[which_set] = CudaNdarray(one_hot_data)
    return _data_cache[which_set]

class PTB(fuel.datasets.Dataset):
    provides_sources = ('features',)
    example_iteration_scheme = None

    def __init__(self, which_set, length, augment=False):
        self.which_set = which_set
        self.length = length
        self.augment = augment
        self.data = get_data(which_set)
        self.num_examples = int(len(self.data) / self.length)
        if self.augment:
            # -1 so we have one self.length worth of room for augmentation
            self.num_examples -= 1
        super(PTB, self).__init__()

    def open(self):
        offset = 0
        if self.augment:
            # choose an offset to get some data augmentation by not always chopping
            # the examples at the same point.
            offset = np.random.randint(self.length)
        # none of this should copy
        data = self.data[offset:]
        # reshape to nonoverlapping examples
        data = (data[:self.num_examples * self.length]
                .reshape((self.num_examples, self.length, self.data.shape[1])))
        # return the data so we will get it as the "state" argument to get_data
        return data

    def get_data(self, state, request):
        if isinstance(request, (tuple, list)):
            request = np.array(request, dtype=np.int64)
            return (state.take(request, 0),)
        return (state[request],)

def get_stream(which_set, batch_size, length, num_examples=None, augment=False):
    dataset = PTB(which_set, length=length, augment=augment)
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

class Parameters(object):
    pass

class BatchNormalization(object):
    def __init__(self, shape, initial_gamma=1, initial_beta=0, name=None, use_bias=True, epsilon=1e-5):
        self.shape = shape
        self.initial_gamma = initial_gamma
        self.initial_beta = initial_beta
        self.name = name
        self.use_bias = use_bias
        self.epsilon = epsilon

    @property
    def parameters(self):
        if not hasattr(self, "_parameters"):
            self._parameters = self.allocate_parameters()
        return self._parameters

    def allocate_parameters(self):
        parameters = Parameters()
        for parameter in [
            theano.shared(self.initial_gamma * ones(self.shape), name="gammas"),
            theano.shared(self.initial_beta  * ones(self.shape), name="betas")]:
            add_role(parameter, PARAMETER)
            setattr(parameters, parameter.name, parameter)
            if self.name:
                parameter.name = "%s.%s" % (self.name, parameter.name)
        return parameters

    def construct_graph(self, x, baseline=False, mean=None, var=None):
        p = self.parameters
        assert x.ndim == 2
        mean = x.mean(axis=0) if mean is None else mean
        var  = x.var (axis=0) if var  is None else var
        assert mean.ndim == 1
        assert var.ndim == 1
        betas = p.betas if self.use_bias else 0
        if baseline:
            y = x + betas
        else:
            y = theano.tensor.nnet.bn.batch_normalization(
                inputs=x,
                gamma=p.gammas, beta=betas,
                mean=T.shape_padleft(mean),
                std=T.shape_padleft(T.sqrt(var + self.epsilon)))
        return y, mean, var

class LSTM(object):
    def __init__(self, args, nclasses):
        self.num_hidden = args.num_hidden
        self.initializer = args.initializer
        self.identity_hh = args.initialization == "identity"
        self.peepholes = args.peepholes
        self.nclasses = nclasses
        self.activation = activations[args.activation]

        self.bn_a = BatchNormalization((4 * args.num_hidden,), initial_gamma=args.initial_gamma, name="bn_a", epsilon=args.epsilon)
        self.bn_b = BatchNormalization((4 * args.num_hidden,), initial_gamma=args.initial_gamma, name="bn_b", epsilon=args.epsilon, use_bias=False)
        self.bn_p = BatchNormalization((3 * args.num_hidden,), initial_gamma=args.initial_gamma, name="bn_p", epsilon=args.epsilon, use_bias=False)
        self.bn_c = BatchNormalization((    args.num_hidden,), initial_gamma=args.initial_gamma, name="bn_c", epsilon=args.epsilon)

    @property
    def parameters(self):
        if not hasattr(self, "_parameters"):
            self._parameters = self.allocate_parameters()
        return self._parameters

    def allocate_parameters(self):
        parameters = Parameters()
        Wa = self.initializer((self.num_hidden, 4 * self.num_hidden))

        if self.identity_hh:
            Wa[:self.num_hidden, :self.num_hidden] = np.eye(self.num_hidden)

        for parameter in [
                theano.shared(zeros((self.num_hidden,)), name="h0"),
                theano.shared(zeros((self.num_hidden,)), name="c0"),
                theano.shared(Wa, name="Wa"),
                theano.shared(self.initializer((self.nclasses,   4 * self.num_hidden)), name="Wx"),
                theano.shared(self.initializer((self.num_hidden, 3 * self.num_hidden)), name="Wp"),
                theano.shared(self.initializer((self.nclasses, self.num_hidden)), name="Wsummarize")]:
            add_role(parameter, PARAMETER)
            setattr(parameters, parameter.name, parameter)

        # forget gate bias initialization
        ab_betas = self.bn_a.parameters.betas
        pffft = ab_betas.get_value()
        pffft[self.num_hidden:2*self.num_hidden] = 1.
        ab_betas.set_value(pffft)

        return parameters

    def construct_graph(self, args, x, length, popstats=None):
        p = self.parameters

        # use `symlength` where we need to be able to adapt to longer sequences
        # than the ones we trained on
        symlength = x.shape[0]
        t = T.cast(T.arange(symlength), "int16")
        long_sequence_is_long = T.ge(T.cast(T.arange(symlength), theano.config.floatX), length)
        batch_size = x.shape[1]
        dummy_states = dict(h=T.zeros((symlength, batch_size, args.num_hidden)),
                            c=T.zeros((symlength, batch_size, args.num_hidden)))

        summary = T.dot(x, p.Wsummarize).mean(axis=0) if args.summarize else 0

        output_names = "h c atilde btilde".split()
        for key in "abcp":
            for stat in "mean var".split():
                output_names.append("%s_%s" % (key, stat))

        def stepfn(t, long_sequence_is_long, x, dummy_h, dummy_c, h, c):
            # population statistics are sequences, but we use them
            # like a non-sequence and index it ourselves. this allows
            # us to generalize to longer sequences, in which case we
            # repeat the last element.
            popstats_by_key = dict()
            for key in "abcp":
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

            atilde, btilde, ptilde = T.dot(h, p.Wa), T.dot(x, p.Wx), T.dot(c, p.Wp)
            a_normal, a_mean, a_var = self.bn_a.construct_graph(atilde, baseline=args.baseline, **popstats_by_key["a"])
            b_normal, b_mean, b_var = self.bn_b.construct_graph(btilde, baseline=args.baseline, **popstats_by_key["b"])
            p_normal, p_mean, p_var = self.bn_p.construct_graph(ptilde, baseline=args.baseline, **popstats_by_key["p"])

            # peepholes go only to gates, not to g
            p_normal = T.concatenate([T.zeros((p_normal.shape[0], self.num_hidden)), p_normal], axis=1)
            if not self.peepholes:
                # if we leave p_normal out of the graph, blocks will
                # still consider p.Wp a parameter as it is used in
                # scan outputs, but since it's not part of the graph
                # computing the cost T.grad will complain
                p_normal *= 0

            ab = a_normal + b_normal + p_normal

            g, f, i, o = [fn(ab[:, j * args.num_hidden:(j + 1) * args.num_hidden])
                          for j, fn in enumerate([self.activation] + 3 * [T.nnet.sigmoid])]

            c = dummy_c + f * c + i * g

            c_normal, c_mean, c_var = self.bn_c.construct_graph(c, baseline=args.baseline, **popstats_by_key["c"])

            h = dummy_h + o * self.activation(c_normal)

            return [locals()[name] for name in output_names]

        sequences = [t, long_sequence_is_long, x, dummy_states["h"], dummy_states["c"]]
        outputs_info = [
            T.repeat(p.h0[None, :], batch_size, axis=0) + summary,
            T.repeat(p.c0[None, :], batch_size, axis=0),
        ]
        outputs_info.extend([None] * (len(output_names) - len(outputs_info)))

        outputs, updates = theano.scan(
            stepfn,
            sequences=sequences,
            outputs_info=outputs_info)
        outputs = dict(zip(output_names, outputs))

        if not args.baseline and not args.use_population_statistics:
            # prepare population statistic estimation
            popstats = dict()
            alpha = 0.05
            for key, size in zip("abcp", [4*args.num_hidden, 4*args.num_hidden, args.num_hidden, 3*args.num_hidden]):
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
    if args.initialization in "identity orthogonal".split():
        args.initializer = orthogonal
    elif args.initialization == "uniform":
        args.initializer = lambda shape: uniform(shape, 0.02)
    elif args.initialization == "glorot":
        args.initializer = glorot

    Wy = theano.shared(args.initializer((args.num_hidden, nclasses)), name="Wy")
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
    lstm = LSTM(args, nclasses)
    (outputs, training_updates, dummy_states, popstats) = lstm.construct_graph(
        args, x, length)
    training_graph, training_extensions = construct_common_graph("training", args, outputs, dummy_states, Wy, by, y)
    args.use_population_statistics = True
    (outputs, inference_updates, dummy_states, _) = lstm.construct_graph(
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--num-hidden", type=int, default=1000)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--initialization", choices="identity glorot orthogonal uniform".split(), default="identity")
    parser.add_argument("--initial-gamma", type=float, default=1e-1)
    parser.add_argument("--initial-beta", type=float, default=0)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--activation", choices=list(activations.keys()), default="tanh")
    parser.add_argument("--peepholes", action="store_true")
    parser.add_argument("--optimizer", choices="sgdmomentum rmsprop adam".split(), default="rmsprop")
    parser.add_argument("--learning-rate-decay", type=float, default=0.0)
    parser.add_argument("--summarize", action="store_true")
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
    if args.optimizer == "adam":
        optimizer = Adam(learning_rate=args.learning_rate)
        # zzzz
        optimizer.learning_rate = theano.shared(np.asarray(optimizer.learning_rate, dtype=theano.config.floatX))
    elif args.optimizer == "rmsprop":
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

    extensions.append(SharedVariableModifier(
        optimizer.learning_rate,
        functools.partial(learning_rate_decayer, args.learning_rate_decay)))

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
        ([param.norm(2).copy(name="parameter.norm:%s" % name)
          for name, param in model.get_parameter_dict().items()]
         + [optimizer.learning_rate.copy(name="learning_rate")]),
        data_stream=None, after_epoch=True))

    # performance monitor
    for situation in "training inference".split():
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
        data_stream=get_stream(which_set="train", batch_size=args.batch_size, length=args.length, augment=True),
        algorithm=algorithm, extensions=extensions, model=model)
    main_loop.run()
