#credits to Dima(https://github.com/pbrakel/speech_project/blob/master/initialization.py)
import numpy
import theano

from blocks.initialization import NdarrayInitialization

class NormalizedInitialization(NdarrayInitialization):
    """Initialize parameters with Glorot method.
    Notes
    -----
    For details see
    Understanding the difficulty of training deep feedforward neural networks,
    Glorot, Bengio, 2010
    """
    def generate(self, rng, shape):
        # In the case of diagonal matrix, we initialize the diagonal
        # to zero. This may happen in LSTM for the weights from cell
        # to gates.
        if len(shape) == 1:
            m = numpy.zeros(shape=shape)
        else:
            input_size, output_size = shape
            high = numpy.sqrt(6.) / numpy.sqrt(input_size + output_size)
            m = rng.uniform(-high, high, size=shape)
        return m.astype(theano.config.floatX)

class ConvInitialization(NdarrayInitialization):
    """Initialize weights in convolutonal nets.
       Notes
       -----
       For details see
       http://www.deeplearning.net/tutorial/lenet.html#lenet.
    """
    def generate(self, rng, shape):
        w_bound = 1. / numpy.sqrt(numpy.prod(shape[1:]))
        m = rng.uniform(-w_bound, w_bound, size=shape)
        return m.astype(theano.config.floatX)
