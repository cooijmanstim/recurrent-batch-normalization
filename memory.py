import sys
import logging
from collections import OrderedDict
import numpy as np
import theano, theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import blocks.config
import fuel.datasets, fuel.streams, fuel.transformers, fuel.schemes

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


class Memory(fuel.datasets.Dataset):
    provides_sources = ('x', 'y')
    example_iteration_scheme = None

    def __init__(self, min_interval, max_interval, seed):
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.seed = seed
        super(Memory, self).__init__()

    def open(self):
        return np.random.RandomState(self.seed)

    def get_data(self, state=None, request=None):
        if request is not None:
            raise ValueError

        # the sequence to be remembered
        targets = state.randint(8, size=10)

        interval = state.random_integers(self.min_interval, self.max_interval)

        x = np.zeros((interval + 20, 10), dtype=np.float32)
        x[range(10), targets] = 1.
        x[10:-11, 8] = 1.
        x[-11, 9] = 1.
        x[-10:, 8] = 1.

        y = np.zeros((interval + 20, 10), dtype=np.float32)
        y[:-10, 8] = 1.
        y[range(-10, -0), targets] = 1.

        return (x, y)

def get_stream(min_interval, max_interval, which_set, batch_size, num_examples=10000):
    seed = dict(train=1, valid=2, test=3)[which_set]
    dataset = Memory(min_interval, max_interval, seed)
    stream = fuel.streams.DataStream.default_stream(dataset)
    stream = fuel.transformers.Padding(
        fuel.transformers.Batch(
            stream,
            fuel.schemes.ConstantScheme(batch_size, num_examples)))
    return stream


def construct_rnn(args, x, activation):
    parameters = []

    h0 = theano.shared(zeros((args.num_hidden,)), name="h0")
    Wh = theano.shared((0.99 if args.baseline else 1) * np.eye(args.num_hidden, dtype=theano.config.floatX), name="Wh")
    Wx = theano.shared(orthogonal((10, args.num_hidden)), name="Wx")

    parameters.extend([h0, Wh, Wx])

    gammas = theano.shared(args.initial_gamma * ones((args.num_hidden,)), name="gammas")
    betas  = theano.shared(args.initial_beta  * ones((args.num_hidden,)), name="betas")

    if args.baseline:
        parameters.extend([betas])
        def bn(x, gammas, betas):
            return x + betas
    else:
        parameters.extend([gammas, betas])
        def bn(x, gammas, betas):
            mean, var = x.mean(axis=0, keepdims=True), x.var(axis=0, keepdims=True)
            # if only
            mean.tag.batchstat, var.tag.batchstat = True, True
            #var = T.maximum(var, args.epsilon)
            var = var + args.epsilon
            return (x - mean) / T.sqrt(var) * gammas + betas

    xtilde = T.dot(x, Wx)

    if args.noise:
        # prime h with white noise
        Trng = MRG_RandomStreams()
        h_prime = Trng.normal((xtilde.shape[1], args.num_hidden), std=args.noise)
    elif args.summarize:
        # prime h with mean of example
        h_prime = x.mean(axis=[0, 2])[:, None]
    else:
        h_prime = 0

    dummy_states = dict(h     =T.zeros((xtilde.shape[0], xtilde.shape[1], args.num_hidden)),
                        htilde=T.zeros((xtilde.shape[0], xtilde.shape[1], args.num_hidden)))

    def stepfn(xtilde, dummy_h, dummy_htilde, h):
        htilde = dummy_htilde + T.dot(h, Wh) + xtilde
        h = dummy_h + activation(bn(htilde, gammas, betas))
        return h, htilde

    [h, htilde], _ = theano.scan(stepfn,
                                 sequences=[xtilde, dummy_states["h"], dummy_states["htilde"]],
                                 outputs_info=[T.repeat(h0[None, :], xtilde.shape[1], axis=0) + h_prime,
                                               None])

    return dict(h=h, htilde=htilde), dummy_states, parameters

def construct_lstm(args, x, activation):
    parameters = []

    h0 = theano.shared(zeros((args.num_hidden,)), name="h0")
    c0 = theano.shared(zeros((args.num_hidden,)), name="c0")
    Wa = theano.shared(np.concatenate([
        np.eye(args.num_hidden),
        orthogonal((args.num_hidden, 3 * args.num_hidden)),
    ], axis=1).astype(theano.config.floatX), name="Wa")
    Wx = theano.shared(orthogonal((10, 4 * args.num_hidden)), name="Wx")

    parameters.extend([h0, c0, Wa, Wx])

    a_gammas = theano.shared(args.initial_gamma * ones((4 * args.num_hidden,)), name="a_gammas")
    b_gammas = theano.shared(args.initial_gamma * ones((4 * args.num_hidden,)), name="b_gammas")
    ab_betas = theano.shared(args.initial_beta  * ones((4 * args.num_hidden,)), name="ab_betas")
    h_gammas = theano.shared(args.initial_gamma * ones((args.num_hidden,)), name="h_gammas")
    h_betas  = theano.shared(args.initial_beta  * ones((args.num_hidden,)), name="h_betas")

    # forget gate bias initialization
    pffft = ab_betas.get_value()
    pffft[args.num_hidden:2*args.num_hidden] = 1.
    ab_betas.set_value(pffft)

    if args.baseline:
        parameters.extend([ab_betas, h_betas])
        def bn(x, gammas, betas):
            return x + betas
    else:
        parameters.extend([a_gammas, b_gammas, h_gammas, ab_betas, h_betas])
        def bn(x, gammas, betas):
            mean, var = x.mean(axis=0, keepdims=True), x.var(axis=0, keepdims=True)
            # if only
            mean.tag.batchstat, var.tag.batchstat = True, True
            #var = T.maximum(var, args.epsilon)
            var = var + args.epsilon
            return (x - mean) / T.sqrt(var) * gammas + betas

    xtilde = T.dot(x, Wx)

    if args.noise:
        # prime h with white noise
        Trng = MRG_RandomStreams()
        h_prime = Trng.normal((xtilde.shape[1], args.num_hidden), std=args.noise)
    elif args.summarize:
        # prime h with mean of example
        h_prime = x.mean(axis=[0, 2])[:, None]
    else:
        h_prime = 0

    dummy_states = dict(h=T.zeros((xtilde.shape[0], xtilde.shape[1], args.num_hidden)),
                        c=T.zeros((xtilde.shape[0], xtilde.shape[1], args.num_hidden)))

    def stepfn(xtilde, dummy_h, dummy_c, h, c):
        atilde = T.dot(h, Wa)
        btilde = xtilde
        a = bn(atilde, a_gammas, ab_betas)
        b = bn(btilde, b_gammas, 0)
        ab = a + b
        g, f, i, o = [fn(ab[:, j * args.num_hidden:(j + 1) * args.num_hidden])
                      for j, fn in enumerate([activation] + 3 * [T.nnet.sigmoid])]
        c = dummy_c + f * c + i * g
        htilde = c
        h = dummy_h + o * activation(bn(htilde, h_gammas, h_betas))
        return h, c, atilde, btilde, htilde

    [h, c, atilde, btilde, htilde], _ = theano.scan(
        stepfn,
        sequences=[xtilde, dummy_states["h"], dummy_states["c"]],
        outputs_info=[T.repeat(h0[None, :], xtilde.shape[1], axis=0) + h_prime,
                      T.repeat(c0[None, :], xtilde.shape[1], axis=0),
                      None, None, None])
    return dict(h=h, c=c, atilde=atilde, btilde=btilde, htilde=htilde), dummy_states, parameters

if __name__ == "__main__":
    nclasses = 10
    batch_size = 100

    activations = dict(
        tanh=T.tanh,
        identity=lambda x: x,
        relu=lambda x: T.max(0, x))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--num-hidden", type=int, default=100)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--lstm", action="store_true")
    parser.add_argument("--initial-gamma", type=float, default=0.25)
    parser.add_argument("--initial-beta", type=float, default=0)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--activation", choices=list(activations.keys()), default="tanh")
    parser.add_argument("--continue-from")
    parser.add_argument("--interval", type=int, default=100)
    args = parser.parse_args()

    assert not (args.noise and args.summarize)

    np.random.seed(args.seed)
    blocks.config.config.default_seed = args.seed

    if args.continue_from:
        from blocks.serialization import load
        main_loop = load(args.continue_from)
        main_loop.run()
        sys.exit(0)


    data_dbg = next(get_stream(which_set="valid",
                               min_interval=args.interval,
                               max_interval=args.interval,
                               batch_size=batch_size).get_epoch_iterator())


    min_sequence_length = args.interval + 20

    constructor = construct_lstm if args.lstm else construct_rnn

    Wy = theano.shared(orthogonal((args.num_hidden, nclasses)), name="Wy")
    by = theano.shared(np.zeros((nclasses,), dtype=theano.config.floatX), name="by")

    ### graph construction
    x, y = T.tensor3("x"), T.tensor3("y")
    x_mask, y_mask = T.matrix("x_mask"), T.matrix("y_mask")
    x.tag.test_value = data_dbg[2]
    y.tag.test_value = data_dbg[0]
    x_mask.tag.test_value = data_dbg[3]
    y_mask.tag.test_value = data_dbg[1]

    mask = x_mask + 0 * y_mask # need to use both in the graph or theano and blocks will complain

    # move time axis before batch axis
    x = x.dimshuffle(1, 0, 2)
    y = y.dimshuffle(1, 0, 2)
    mask = mask.dimshuffle(1, 0)

    states, dummy_states, parameters = constructor(args, x=x, activation=activations[args.activation])

    ytilde = T.dot(states["h"], Wy) + by
    ytilde_reshape = ytilde.reshape((ytilde.shape[0] * ytilde.shape[1], ytilde.shape[2]))
    yhat = T.nnet.softmax(ytilde_reshape).reshape((ytilde.shape[0], ytilde.shape[1], ytilde.shape[2]))
    #yhat = ytilde

    errors = T.neq(T.argmax(y, axis=2), T.argmax(yhat, axis=2)).reshape(mask.shape)
    flat_y = y.reshape((-1, nclasses))
    flat_yhat = yhat.reshape((-1, nclasses))
    cross_entropy = T.nnet.categorical_crossentropy(flat_yhat, flat_y).reshape(mask.shape)

    ## masking
    errors = mask * errors
    cross_entropy = mask * cross_entropy

    # following Amar & Martin we take the mean across time (rather than require all steps to be correct)
    error_rate = errors.mean()
    cross_entropy = cross_entropy.mean()
    cost = cross_entropy

    state_grads = dict((k, T.grad(cost, v)) for k, v in dummy_states.items())

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

    parameters.extend([Wy, by])
    for parameter in parameters:
        add_role(parameter, PARAMETER)

    graph = ComputationGraph(cost)
    step_rule = CompositeRule([
        StepClipping(1.),
        #Momentum(learning_rate=args.learning_rate, momentum=0.9),
        RMSProp(learning_rate=args.learning_rate, decay_rate=0.9),
    ])
    algorithm = GradientDescent(cost=cost, parameters=graph.parameters, step_rule=step_rule)

    model = Model(cost)
    extensions = []

    # step monitor
    step_channels = []
    step_channels.extend([
        algorithm.steps[param].norm(2).copy(name="step_norm:%s" % name)
        for name, param in model.get_parameter_dict().items()])
    step_channels.append(algorithm.total_step_norm.copy(name="total_step_norm"))
    step_channels.append(algorithm.total_gradient_norm.copy(name="total_gradient_norm"))
    step_channels.extend([cross_entropy.copy(name="cross_entropy"),
                          error_rate.copy(name="error_rate")])
    logger.warning("constructing training data monitor")
    extensions.append(TrainingDataMonitoring(
        step_channels, prefix="iteration", after_batch=True))

    # parameter monitor
    extensions.append(DataStreamMonitoring(
        [param.norm(2).copy(name="parameter.norm:%s" % name)
         for name, param in model.get_parameter_dict().items()],
        data_stream=None, after_epoch=True))

    # performance monitor
    for which_set in "train valid test".split():
        logger.warning("constructing %s monitor" % which_set)
        channels = [cross_entropy.copy(name="cross_entropy"),
                    error_rate.copy(name="error_rate")]
        extensions.append(DataStreamMonitoring(
            channels,
            prefix=which_set, after_epoch=True,
            data_stream=get_stream(which_set=which_set,
                                   min_interval=args.interval,
                                   max_interval=args.interval,
                                   batch_size=batch_size)))

    hiddenthingsdumper = DumpVariables("hiddens", graph.inputs,
                                       [v.copy(name="%s%s" % (k, suffix))
                                        for suffix, things in [("", states), ("_grad", state_grads)]
                                        for k, v in things.items()],
                                       batch=next(get_stream(which_set="train",
                                                             min_interval=args.interval,
                                                             max_interval=args.interval,
                                                             batch_size=batch_size)
                                                  .get_epoch_iterator(as_dict=True)),
                                       before_training=True, every_n_epochs=10)

    extensions.extend([
        hiddenthingsdumper,
        TrackTheBest("valid_error_rate", "best_valid_error_rate"),
        DumpBest("best_valid_error_rate", "best.zip"),
        FinishAfter(after_n_epochs=args.num_epochs),
        # validation error improvements are sparse on the memory task
        #FinishIfNoImprovementAfter("best_valid_error_rate", epochs=30),
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
        data_stream=get_stream(which_set="train",
                                   min_interval=args.interval,
                                   max_interval=args.interval,
                                   batch_size=batch_size),
        algorithm=algorithm, extensions=extensions, model=model)

    #from tabulate import tabulate
    #print "parameter sizes:"
    #print tabulate((key, "x".join(map(str, value.get_value().shape)), value.get_value().size)
    #               for key, value in main_loop.model.get_parameter_dict().items())

    main_loop.run()
