import sys
import logging
from collections import OrderedDict
import numpy as np
import util
import theano, theano.tensor as T
import blocks.config
import fuel.datasets, fuel.streams, fuel.transformers, fuel.schemes
from blocks.config import config

logging.basicConfig()
logger = logging.getLogger(__name__)

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

from blocks.extensions import SimpleExtension
from blocks.serialization import secure_dump
class DumpVariables(SimpleExtension):
    def __init__(self, save_path, inputs, variables, batch, **kwargs):
        kwargs.setdefault("before_epoch", True)
        super(DumpVariables, self).__init__(**kwargs)
        self.save_path = save_path
        self.variables = variables
        self.function = theano.function(inputs, variables)
        self.batch = batch
        self.i = 0

    def do(self, which_callback, *args):
        if which_callback == "before_epoch":
            values = dict((variable.name, np.asarray(value)) for variable, value in
                          zip(self.variables, self.function(**self.batch)))
            secure_dump(values, "%s_%i.pkl" % (self.save_path, self.i))
            self.i += 1

if __name__ == "__main__":
    config.recursion_limit = 100000
    sequence_length = 784
    nclasses = 10
    batch_size = 100

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", choices="identity relu tanh linh soft_tanh")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=1)
    parser.add_argument("--noise", type=float, default=1)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--num-hidden", type=int, default=100)
    args = parser.parse_args()

    np.random.seed(args.seed)
    blocks.config.config.default_seed = args.seed

    from blocks.graph import ComputationGraph
    from blocks.algorithms import GradientDescent, RMSProp, StepClipping, CompositeRule, Momentum
    from blocks.model import Model
    from blocks.extensions import FinishAfter, Printing, ProgressBar, Timing
    from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
    from blocks.extensions.stopping import FinishIfNoImprovementAfter
    from blocks.extensions.training import TrackTheBest
    from blocks.extensions.saveload import Checkpoint
    from extensions import DumpLog, DumpBest, PrintingTo
    from blocks.main_loop import MainLoop
    from blocks.utils import shared_floatx_zeros

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

    h0 = theano.shared(np.zeros((args.num_hidden,), dtype=theano.config.floatX), name="h0")
    Wh = theano.shared(np.eye(args.num_hidden, dtype=theano.config.floatX), name="Wh")
    Wx = theano.shared(orthogonal((1, args.num_hidden)), name="Wx")
    Wy = theano.shared(orthogonal((args.num_hidden, nclasses)), name="Wy")
    by = theano.shared(np.zeros((nclasses,), dtype=theano.config.floatX), name="by")

    initial_gamma, initial_beta = 0.25, 0
    if args.share:
        gammas = theano.shared(initial_gamma * np.ones((args.num_hidden,), dtype=theano.config.floatX), name="gammas")
        betas  = theano.shared(initial_beta  * np.ones((args.num_hidden,), dtype=theano.config.floatX), name="betas")
    else:
        gammas = theano.shared(initial_gamma * np.ones((784, args.num_hidden), dtype=theano.config.floatX), name="gammas")
        betas  = theano.shared(initial_beta  * np.ones((784, args.num_hidden), dtype=theano.config.floatX), name="betas")

    parameters = [h0, Wh, Wx, Wy, by, gammas, betas]
    if args.activation == "soft_tanh":
        parameters.extend([Ws, bs])
    from blocks.roles import add_role, PARAMETER
    for parameter in parameters:
        add_role(parameter, PARAMETER)

    x, y = T.tensor4("features"), T.imatrix("targets")

    # make sequential
    x = x.reshape((x.shape[0], sequence_length, 1))
    # remove bogus dimension
    y = y.flatten(ndim=1)

    # move time axis before batch axis
    x = x.dimshuffle(1, 0, 2)

    def bn(x, gamma, beta):
        mean, var = x.mean(axis=0, keepdims=True), x.var(axis=0, keepdims=True)
        #var = T.maximum(var, args.epsilon)
        var = var + args.epsilon
        return (x - mean) / T.sqrt(var) * gamma + beta

    xtilde = T.dot(x, Wx)
    dummy_h = T.zeros((xtilde.shape[0], xtilde.shape[1], args.num_hidden))
    dummy_htilde = T.zeros((xtilde.shape[0], xtilde.shape[1], args.num_hidden))
    Trng = theano.sandbox.rng_mrg.MRG_RandomStreams()

    if args.share:
        # repeat gammas/betas over time so the call to scan is the same in both cases
        gammas, betas = [T.repeat(var[None, :], sequence_length, axis=0)
                         for var in (gammas, betas)]

    def stepfn(xtilde, dummy_h, dummy_htilde, gamma, beta, h):
        htilde = dummy_htilde + T.dot(h, Wh) + xtilde
        h = dummy_h + T.tanh(bn(htilde, gamma, beta))
        return h, htilde
    [h, htilde], _ = theano.scan(stepfn,
                                 sequences=[xtilde, dummy_h, dummy_htilde, gammas, betas],
                                 outputs_info=[h0 + Trng.normal((xtilde.shape[1], nhidden), std=args.noise), None])

    ytilde = T.dot(h[-1], Wy) + by
    yhat = T.nnet.softmax(ytilde)

    errors = T.neq(y, T.argmax(yhat, axis=1))
    cross_entropy = T.nnet.categorical_crossentropy(yhat, y)

    error_rate = errors.mean()
    cross_entropy = cross_entropy.mean()
    cost = cross_entropy

    if dummy_h:
        h_grads = T.grad(cross_entropy, dummy_h)
    if dummy_htilde:
        htilde_grads = T.grad(cross_entropy, dummy_htilde)

    ### optimization algorithm definition
    graph = ComputationGraph(cost)
    step_rule = CompositeRule([
        StepClipping(1.),
        #Momentum(learning_rate=args.learning_rate, momentum=0.9),
        RMSProp(learning_rate=args.learning_rate, decay_rate=0.9),
    ])
    algorithm = GradientDescent(cost=cost, parameters=graph.parameters, step_rule=step_rule)

    model = Model(cost)
    extensions = []

    hidden_norms = []
    hidden_grad_norms = []
    if False:
        timesubsample = 10
        for t in xrange(0, sequence_length, timesubsample):
            if dummy_h:
                hidden_grad_norms.append(h_grads[t, :, :].norm(2).copy(name="grad_norm:h_%03i" % t))
                hidden_grad_norms.append(h_grads[t, :, :].var(axis=0).min().copy(name="grad_minvar:h_%03i" % t))
                hidden_grad_norms.append(h_grads[t, :, :].var(axis=0).max().copy(name="grad_maxvar:h_%03i" % t))
            if dummy_htilde:
                hidden_grad_norms.append(htilde_grads[t, :, :].norm(2).copy(name="grad_norm:htilde_%03i" % t))
                hidden_grad_norms.append(htilde_grads[t, :, :].var(axis=0).min().copy(name="grad_minvar:htilde_%03i" % t))
                hidden_grad_norms.append(htilde_grads[t, :, :].var(axis=0).max().copy(name="grad_maxvar:htilde_%03i" % t))
            hidden_norms.append(h[t, :, :].norm(2).copy(name="norm:h_%03i" % t))
            hidden_norms.append(h[t, :, :].var(axis=0).min().copy(name="minvar:h_%03i" % t))
            hidden_norms.append(h[t, :, :].var(axis=0).max().copy(name="maxvar:h_%03i" % t))
            hidden_norms.append(htilde[t, :, :].norm(2).copy(name="norm:htilde_%03i" % t))
            hidden_norms.append(htilde[t, :, :].var(axis=0).min().copy(name="minvar:htilde_%03i" % t))
            hidden_norms.append(htilde[t, :, :].var(axis=0).max().copy(name="maxvar:htilde_%03i" % t))

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
        step_channels + hidden_norms + hidden_grad_norms, prefix="iteration", after_batch=True))

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
            data_stream=get_stream(which_set=which_set, batch_size=batch_size, num_examples=1000)))

    hiddenthingsdumper = DumpVariables("hiddens", graph.inputs,
                                       [h.copy(name="h"),
                                        h_grads.copy(name="h_grads"),
                                        htilde.copy(name="htilde"),
                                        htilde_grads.copy(name="htilde_grads")],
                                       batch=next(get_stream(which_set="train",
                                                             batch_size=batch_size,
                                                             num_examples=batch_size)
                                                  .get_epoch_iterator(as_dict=True)))

    extensions.extend([
        hiddenthingsdumper,
        TrackTheBest("valid_error_rate", "best_valid_error_rate"),
        DumpBest("best_valid_error_rate", "best.zip"),
        FinishAfter(after_n_epochs=args.num_epochs),
        # validation error improvements are sparse on the memory task
        #FinishIfNoImprovementAfter("best_valid_error_rate", epochs=30),
        Checkpoint("checkpoint.zip", on_interrupt=False, every_n_epochs=1, use_cpickle=True),
        DumpLog("log.pkl", after_epoch=True),
        ProgressBar(),
        Timing(),
        Printing(),
        PrintingTo("log"),
    ])
    main_loop = MainLoop(
        data_stream=get_stream(which_set="train", batch_size=batch_size),
        algorithm=algorithm, extensions=extensions, model=model)

    #from tabulate import tabulate
    #print "parameter sizes:"
    #print tabulate((key, "x".join(map(str, value.get_value().shape)), value.get_value().size)
    #               for key, value in main_loop.model.get_parameter_dict().items())

    main_loop.run()
