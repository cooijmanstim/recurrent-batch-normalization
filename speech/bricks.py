import numpy

import theano

from theano import config, shared, tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams

from blocks.bricks import application, Initializable, lazy, Brick, Identity, Linear
from blocks.bricks.recurrent import BaseRecurrent, recurrent, SimpleRecurrent, LSTM
from blocks.initialization import Constant
from blocks.roles import add_role, WEIGHT, BIAS, INITIAL_STATE, PARAMETER
from blocks.utils import shared_floatx_nans, shared_floatx_zeros

from initialization import NormalizedInitialization


class SimpleRecurrentBatchNorm(BaseRecurrent, Initializable):
    """The traditional recurrent transition.
    The most well-known recurrent transition: a matrix multiplication,
    Parameters
    rec1 = TimLSTM(not batch_norm, inp
    ----------
    dim : int
        The dimension of the hidden state
    activation : :class:`.Brick`
        The brick to apply as activation.
    Notes
    -----
    See :class:`.Initializable` for initialization parameters.
    """
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation, seq_len, epsilon=1e-6, **kwargs):
        super(SimpleRecurrentBatchNorm, self).__init__(**kwargs)
        self.dim = dim
        self.children = [activation]
        self.epsilon = epsilon
        self.seq_len = seq_len
        # Keeping track of the means and variances during the training.
        means_val = numpy.zeros((seq_len, self.dim), dtype=config.floatX)
        self.pop_means = shared(name='means', value=means_val)
        vars_val = numpy.ones((seq_len, self.dim), dtype=config.floatX)
        self.pop_vars = shared(name='varainces', value=vars_val)

    @property
    def W(self):
        return self.parameters[0]
    
    @property
    def gamma(self):
        return self.parameters[1]
    
    @property
    def beta(self):
        return self.parameters[2]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in (SimpleRecurrentBatchNorm._training_rec_apply.sequences +
                    SimpleRecurrentBatchNorm._training_rec_apply.states):
            return self.dim
        if name in ['gamma', 'beta']:
            return self.dim
        return super(SimpleRecurrentBatchNorm, self).get_dim(name)

    def _allocate(self):
        # W
        self.parameters.append(shared_floatx_nans((self.dim, self.dim), name="W"))
        add_role(self.parameters[0], WEIGHT)
        # gamma
        gamma_val = 0.1 * numpy.ones((self.dim), dtype=config.floatX)
        gamma_ = shared(name='gamma', value=gamma_val)
        add_role(gamma_, PARAMETER)
        self.parameters.append(gamma_)
        # beta
        beta_val = numpy.zeros((self.dim), dtype=config.floatX)
        beta_ = shared(name='beta', value=beta_val)
        add_role(beta_, PARAMETER)
        self.parameters.append(beta_)
        #self.parameters.append(shared_floatx_zeros((self.dim,),
        #                                           name="initial_state"))
        #add_role(self.parameters[3], INITIAL_STATE)

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        Trng = MRG_RandomStreams()
        return Trng.normal((batch_size, self.dim), std=1.0)
        #return tensor.repeat(self.parameters[3][None, :], batch_size, 0)

    @initial_states.property('outputs')
    def initial_states_outputs(self):
        return self._training_rec_apply.states

    def _initialize(self):
        self.weights_init.initialize(self.W, self.rng)

    #def get_replacements(self):
    #    """Returns the replacements for the computation graph."""
    #    return {self.training_output: self.inference_output}

    #def get_updates(self, n_batches):
    #    """Update the population means and variances of the brick. Use
    #       n_batches from the training dataset to do so.
    #    
    #    """
    #    m_u = (self.pop_means, (self.pop_means
    #                            + 1./n_batches * self.batch_means))
    #    v_u = (self.pop_vars, (self.pop_vars
    #                           + 1./n_batches * self.batch_vars))
    #    return [m_u, v_u]

    @application(inputs=['inputs', 'mask'], outputs=['output'])
    def apply(self, inputs=None, mask=None):
        train_out = self._training_rec_apply(inputs=inputs, mask=mask)
        self.training_output = train_out[0]
        self.batch_means = train_out[1]
        self.batch_vars = train_out[2]
        self.pre_activation = train_out[3]
        inf_out = self._inference_rec_apply(inputs=inputs, mask=mask,
                                            pop_means=self.pop_means,
                                            pop_vars=self.pop_vars)
        self.inference_output = inf_out
        return self.training_output

    @recurrent(sequences=['inputs', 'mask'],
               states=['states'], contexts=[],
               outputs=['states', 'batch_means', 'batch_vars', 'pre_activation'])
    def _training_rec_apply(self, inputs=None, states=None, mask=None):
        next_states = inputs + tensor.dot(states, self.W)
        batch_means = next_states.mean(axis=0, keepdims=False,
                                       dtype=config.floatX)
        batch_vars = next_states.var(axis=0, keepdims=False)
        next_states -= batch_means.dimshuffle('x', 0)
        next_states /= tensor.sqrt(batch_vars.dimshuffle('x', 0)
                                   + self.epsilon)
        next_states = self.gamma*next_states + self.beta
        pre_activation = next_states
        next_states = self.children[0].apply(next_states)
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states, batch_means, batch_vars, pre_activation
    
    @recurrent(sequences=['inputs', 'mask', 'pop_means', 'pop_vars'],
               states=['states'], outputs=['states'], contexts=[])
    def _inference_rec_apply(self, inputs=None, states=None, mask=None,
                             pop_means=None, pop_vars=None):
        next_states = inputs + tensor.dot(states, self.W)
        next_states -= pop_means
        next_states /= tensor.sqrt(pop_vars + self.epsilon)
        next_states = self.gamma*next_states + self.beta
        next_states = self.children[0].apply(next_states)
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states


class LSTMBatchNorm(BaseRecurrent, Initializable):
    u"""Long Short Term Memory.
    Every unit of an LSTM is equipped with input, forget and output gates.
    This implementation is based on code by Mohammad Pezeshki that
    implements the architecture used in [GSS03]_ and [Grav13]_. It aims to
    do as many computations in parallel as possible and expects the last
    dimension of the input to be four times the output dimension.
    Unlike a vanilla LSTM as described in [HS97]_, this model has peephole
    connections from the cells to the gates. The output gates receive
    information about the cells at the current time step, while the other
    gates only receive information about the cells at the previous time
    step. All 'peephole' weight matrices are diagonal.
    .. [GSS03] Gers, Felix A., Nicol N. Schraudolph, and Juergen
        Schmidhuber, *Learning precise timing with LSTM recurrent
        networks*, Journal of Machine Learning Research 3 (2003),
        pp. 115-143.
    .. [Grav13] Graves, Alex, *Generating sequences with recurrent neural
        networks*, arXiv preprint arXiv:1308.0850 (2013).
    .. [HS97] Sepp Hochreiter, and Juergen Schmidhuber, *Long Short-Term
        Memory*, Neural Computation 9(8) (1997), pp. 1735-1780.
    Parameters
    ----------
    dim : int
        The dimension of the hidden state.
    activation : :class:`.Brick`, optional
        The activation function. The default and by far the most popular
        is :class:`.Tanh`.
    Notes
    -----
    See :class:`.Initializable` for initialization parameters.
    """
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, epsilon=1e-6, **kwargs):
        super(LSTMBatchNorm, self).__init__(**kwargs)
        self.dim = dim
        self.epsilon = epsilon

        if not activation:
            activation = Tanh()
        self.children = [activation]

    def get_dim(self, name):
        if name == 'inputs':
            return self.dim * 4
        if name in ['states', 'cells']:
            return self.dim
        if name == 'mask':
            return 0
        return super(LSTMBatchNorm, self).get_dim(name)

    def _allocate(self):
        self.W_state = shared_floatx_nans((self.dim, 4*self.dim),
                                          name='W_state')
        self.W_cell_to_in = shared_floatx_nans((self.dim,),
                                               name='W_cell_to_in')
        self.W_cell_to_forget = shared_floatx_nans((self.dim,),
                                                   name='W_cell_to_forget')
        self.W_cell_to_out = shared_floatx_nans((self.dim,),
                                                name='W_cell_to_out')
        # The underscore is required to prevent collision with
        # the `initial_state` application method
        self.initial_state_ = shared_floatx_zeros((self.dim,),
                                                  name="initial_state")
        self.initial_cells = shared_floatx_zeros((self.dim,),
                                                 name="initial_cells")
        add_role(self.W_state, WEIGHT)
        add_role(self.W_cell_to_in, WEIGHT)
        add_role(self.W_cell_to_forget, WEIGHT)
        add_role(self.W_cell_to_out, WEIGHT)
        add_role(self.initial_state_, INITIAL_STATE)
        add_role(self.initial_cells, INITIAL_STATE)
        
        # gamma
        gamma_val = 0.1 * numpy.ones((self.dim), dtype=config.floatX)
        self.gamma = shared(name='gamma', value=gamma_val)
        add_role(self.gamma, PARAMETER)
        # beta
        beta_val = numpy.zeros((self.dim), dtype=config.floatX)
        self.beta = shared(name='beta', value=beta_val)
        add_role(self.beta, PARAMETER)

        self.parameters = [
            self.W_state, self.W_cell_to_in, self.W_cell_to_forget,
            self.W_cell_to_out, self.initial_state_, self.initial_cells,
            self.gamma, self.beta]

    def _initialize(self):
        for weights in self.parameters[:4]:
            self.weights_init.initialize(weights, self.rng)

    @recurrent(sequences=['inputs', 'mask'], states=['states', 'cells'],
               contexts=[], outputs=['states', 'cells'])
    def apply(self, inputs, states, cells, mask=None):
        """Apply the Long Short Term Memory transition.
        Parameters
        ----------
        states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current states in the shape
            (batch_size, features). Required for `one_step` usage.
        cells : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current cells in the shape
            (batch_size, features). Required for `one_step` usage.
        inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs in the shape (batch_size,
            features * 4). The `inputs` needs to be four times the
            dimension of the LSTM brick to insure each four gates receive
            different transformations of the input. See [Grav13]_
            equations 7 to 10 for more details. The `inputs` are then split
            in this order: Input gates, forget gates, cells and output
            gates.
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            data available, 0 if not. Assumed to be 1-s only if not given.
        .. [Grav13] Graves, Alex, *Generating sequences with recurrent
            neural networks*, arXiv preprint arXiv:1308.0850 (2013).
        Returns
        -------
        states : :class:`~tensor.TensorVariable`
            Next states of the network.
        cells : :class:`~tensor.TensorVariable`
            Next cell activations of the network.
        """
        def slice_last(x, no):
            return x[:, no*self.dim: (no+1)*self.dim]

        nonlinearity = self.children[0].apply

        activation = tensor.dot(states, self.W_state) + inputs
        in_gate = tensor.nnet.sigmoid(slice_last(activation, 0) +
                                      cells * self.W_cell_to_in)
        forget_gate = tensor.nnet.sigmoid(slice_last(activation, 1) +
                                          cells * self.W_cell_to_forget)
        next_cells = (forget_gate * cells +
                      in_gate * nonlinearity(slice_last(activation, 2)))
        out_gate = tensor.nnet.sigmoid(slice_last(activation, 3) +
                                       next_cells * self.W_cell_to_out)
        # Batch Norm
        batch_means = next_cells.mean(axis=0, keepdims=False,
                                      dtype=config.floatX)
        batch_vars = next_cells.var(axis=0, keepdims=False)
        next_cells -= batch_means.dimshuffle('x', 0)
        next_cells /= tensor.sqrt(batch_vars.dimshuffle('x', 0)
                                  + self.epsilon)
        next_cells = self.gamma*next_cells + self.beta
        
        next_states = out_gate * nonlinearity(next_cells)

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
            next_cells = (mask[:, None] * next_cells +
                          (1 - mask[:, None]) * cells)

        return next_states, next_cells

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        #Trng = MRG_RandomStreams()
        #return Trng.normal((batch_size, self.dim), std=1.0)
        return [tensor.repeat(self.initial_state_[None, :], batch_size, 0),
                tensor.repeat(self.initial_cells[None, :], batch_size, 0)]


class MyRecurrent(Brick):
    def __init__(self, recurrent, dims,
                 activations=[Identity(), Identity()], **kwargs):
        super(MyRecurrent, self).__init__(**kwargs)
        self.dims = dims
        self.recurrent = recurrent
        self.activations = activations
        if isinstance(self.recurrent, (SimpleRecurrent, SimpleRecurrentBatchNorm)):
            output_dim = dims[1]
        elif isinstance(self.recurrent, (LSTM, LSTMBatchNorm)):
            output_dim = 4*dims[1]
        else:
            raise NotImplementedError
        self.input_trans = Linear(name='input_trans',
                                  input_dim=dims[0],
                                  output_dim=output_dim,
                                  weights_init=NormalizedInitialization(),
                                  biases_init=Constant(0))
        self.output_trans = Linear(name='output_trans',
                                   input_dim=dims[1],
                                   output_dim=dims[2],
                                   weights_init=NormalizedInitialization(),
                                   biases_init=Constant(0))
        self.children = ([self.input_trans, self.recurrent, self.output_trans]
                         + self.activations)
        
    def _initialize(self):
        self.input_trans.initialize()
        self.output_trans.initialize()
        #self.recurrent.initialize()

    @application
    def apply(self, input_, input_mask=None, *args, **kwargs):
        input_recurrent = self.input_trans.apply(input_)
        try:
            input_recurrent = self.activations[0].apply(input_recurrent, input_mask=input_mask)
        except TypeError:
            input_recurrent = self.activations[0].apply(input_recurrent)
        output_recurrent = self.recurrent.apply(inputs=input_recurrent,
                                                mask=input_mask)
        if isinstance(self.recurrent, (LSTM, LSTMBatchNorm)):
            output_recurrent = output_recurrent[0]
        output = self.output_trans.apply(output_recurrent)
        try:
            output = self.activations[1].apply(output, input_mask=input_mask)
        except TypeError:
            output = self.activations[1].apply(output)
        return output


# ----------------------------------------------------------------------------
# TIM CODE
# ----------------------------------------------------------------------------


def zeros(shape):
    return numpy.zeros(shape, dtype=theano.config.floatX)

def ones(shape):
    return numpy.ones(shape, dtype=theano.config.floatX)

def glorot(shape):
    d = numpy.sqrt(6. / sum(shape))
    return numpy.random.uniform(-d, +d, size=shape).astype(theano.config.floatX)

def orthogonal(shape):
    # taken from https://gist.github.com/kastnerkyle/f7464d98fe8ca14f2a1a
    """ benanne lasagne ortho init (faster than qr approach)"""
    flat_shape = (shape[0], numpy.prod(shape[1:]))
    a = numpy.random.normal(0.0, 1.0, flat_shape)
    u, _, v = numpy.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v  # pick the one with the correct shape
    q = q.reshape(shape)
    return q[:shape[0], :shape[1]].astype(theano.config.floatX)



class TimLSTM(Brick):

    def __init__(self, baseline, input_dim, state_dim, activation, noise=None, initial_gamma=0.1, initial_beta=0.0, epsilon=1e-6, **kwargs):
        super(TimLSTM, self).__init__(**kwargs)
        self.baseline = baseline
        self.children = [activation]
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.noise = noise
        self.initial_gamma = initial_gamma
        self.initial_beta = initial_beta
        self.epsilon = epsilon
        
    def _initialize(self):
        self.parameters = []
        h0 = theano.shared(zeros((self.state_dim,)), name="h0")
        c0 = theano.shared(zeros((self.state_dim,)), name="c0")
        Wa = theano.shared(numpy.concatenate([
            numpy.eye(self.state_dim),
            orthogonal((self.state_dim, 3 * self.state_dim)),
        ], axis=1).astype(theano.config.floatX), name="Wa")
        Wx = theano.shared(orthogonal((self.input_dim, 4 * self.state_dim)), name="Wx")

        self.parameters.extend([h0, c0, Wa, Wx])

        a_gammas = theano.shared(self.initial_gamma * ones((4 * self.state_dim,)), name="a_gammas")
        b_gammas = theano.shared(self.initial_gamma * ones((4 * self.state_dim,)), name="b_gammas")
        ab_betas = theano.shared(self.initial_beta  * ones((4 * self.state_dim,)), name="ab_betas")
        h_gammas = theano.shared(self.initial_gamma * ones((self.state_dim,)), name="h_gammas")
        h_betas  = theano.shared(self.initial_beta  * ones((self.state_dim,)), name="h_betas")

        # forget gate bias initialization
        pffft = ab_betas.get_value()
        pffft[self.state_dim:2*self.state_dim] = 1.
        ab_betas.set_value(pffft)
        if self.baseline:
            self.parameters.extend([ab_betas, h_betas])
        else:
            self.parameters.extend([a_gammas, b_gammas, h_gammas, ab_betas, h_betas])

    def bn(self, x, gammas, betas):
        if self.baseline:
            return x + betas
        else:
            mean, var = x.mean(axis=0, keepdims=True), x.var(axis=0, keepdims=True)
            # if only
            mean.tag.batchstat, var.tag.batchstat = True, True
            #var = tensor.maximum(var, args.epsilon)
            var = var + self.epsilon
            return (x - mean) / tensor.sqrt(var) * gammas + betas

    @application
    def apply(self, x):

        # lazy hack
        h0 = self.parameters[0]
        c0 = self.parameters[1]
        Wa = self.parameters[2]
        Wx = self.parameters[3]
        if self.baseline:
            ab_betas = self.parameters[4]
            h_betas = self.parameters[5]
            a_gammas = None
            b_gammas = None
            h_gammas = None
        else:
            a_gammas = self.parameters[4]
            b_gammas = self.parameters[5]
            h_gammas = self.parameters[6]
            ab_betas = self.parameters[7]
            h_betas = self.parameters[8]

        xtilde = tensor.dot(x, Wx)

        if self.noise:
            # prime h with white noise
            Trng = MRG_RandomStreams()
            h_prime = Trng.normal((xtilde.shape[1], self.state_dim), std=args.noise)
        #elif args.summarize:
        #    # prime h with summary of example
        #    Winit = theano.shared(orthogonal((nclasses, self.state_dim)), name="Winit")
        #    parameters.append(Winit)
        #    h_prime = tensor.dot(x, Winit).mean(axis=0)
        else:
            h_prime = 0

        dummy_states = dict(h=tensor.zeros((xtilde.shape[0], xtilde.shape[1], self.state_dim)),
                            c=tensor.zeros((xtilde.shape[0], xtilde.shape[1], self.state_dim)))

        def stepfn(xtilde, dummy_h, dummy_c, h, c):
            atilde = tensor.dot(h, Wa)
            btilde = xtilde
            a = self.bn(atilde, a_gammas, ab_betas)
            b = self.bn(btilde, b_gammas, 0)
            ab = a + b
            g, f, i, o = [fn(ab[:, j * self.state_dim:(j + 1) * self.state_dim])
                          for j, fn in enumerate([self.children[0].apply] + 3 * [tensor.nnet.sigmoid])]
            c = dummy_c + f * c + i * g
            htilde = c
            h = dummy_h + o * self.children[0].apply(self.bn(htilde, h_gammas, h_betas))
            return h, c, atilde, btilde, htilde

        [h, c, atilde, btilde, htilde], _ = theano.scan(
            stepfn,
            sequences=[xtilde, dummy_states["h"], dummy_states["c"]],
            outputs_info=[tensor.repeat(h0[None, :], xtilde.shape[1], axis=0) + h_prime,
                          tensor.repeat(c0[None, :], xtilde.shape[1], axis=0),
                          None, None, None])
        #return dict(h=h, c=c, atilde=atilde, btilde=btilde, htilde=htilde), dummy_states, parameters
        return h


