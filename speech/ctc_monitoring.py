import logging
import theano

import numpy

from blocks.extensions import SimpleExtension
from blocks.extensions.monitoring import MonitoringExtension
from blocks.search import BeamSearch

from picklable_itertools import groupby

from evaluation import Evaluation, BeamSearchLM
#from language_model import TrigramLanguageModel

logger = logging.getLogger()


class BeamSearchEvaluator(object):
    def __init__(self, eol_symbol, beam_size, x, x_mask, samples,
                 phoneme_dict=None, black_list=None, language_model=False):
        if black_list is None:
            self.black_list = []
        else:
            self.black_list = black_list
        self.x = x
        self.x_mask = x_mask
        self.eol_symbol = eol_symbol
        self.beam_size = beam_size
        if language_model:
            lm = TrigramLanguageModel()
            ind_to_word = dict(enumerate(lm.unigrams))
            self.beam_search = BeamSearchLM(
                lm, 1., ind_to_word, beam_size, samples)
        else:
            self.beam_search = BeamSearch(beam_size, samples)
        self.beam_search.compile()
        self.phoneme_dict = phoneme_dict

    def evaluate(self, data_stream, train=False, file_pred=None,
                 file_targets=None):
        loss = 0.
        num_examples = 0
        iterator = data_stream.get_epoch_iterator()
        if train:
            print 'Train evaluation started'
        i = 0
        for inputs in iterator:
            inputs = dict(zip(data_stream.sources, inputs))
            x_mask_val = inputs['features_mask']
            x_val = inputs['features']
            y_val = inputs['phonemes']
            y_mask_val = inputs['phonemes_mask']
            for batch_ind in xrange(inputs['features'].shape[1]):
                if x_val.ndim == 2:
                    input_beam = numpy.tile(x_val[:, batch_ind][:, None],
                        (1, self.beam_size))
                else:
                    input_beam = numpy.tile(x_val[:, batch_ind, :][:, None, :],
                                            (1, self.beam_size, 1))
                input_mask_beam = numpy.tile(x_mask_val[:, batch_ind][:, None],
                                             (1, self.beam_size))
                predictions, _ = self.beam_search.search(
                    {self.x: input_beam,
                     self.x_mask: input_mask_beam},
                    self.eol_symbol, 100)
                predictions = [self.phoneme_dict[phone_ind] for phone_ind
                             in predictions[0]
                             if self.phoneme_dict[phone_ind] not in
                             self.black_list][1:-1]

                targets = y_val[:sum(y_mask_val[:, batch_ind]), batch_ind]
                targets = [self.phoneme_dict[phone_ind] for phone_ind
                             in targets
                             if self.phoneme_dict[phone_ind] not in
                             self.black_list][1:-1]
                predictions = [x[0] for x in groupby(predictions)]
                targets = [x[0] for x in groupby(targets)]
                i += 1
                if file_pred:
                    file_pred.write(' '.join(predictions) + '(%d)\n' % i)
                if file_targets:
                    file_targets.write(' '.join(targets) + '(%d)\n' %i)

                loss += Evaluation.wer([predictions], [targets])
                num_examples += 1

            print '.. found sequence example:', ' '.join(predictions)
            print '.. real output was:       ', ' '.join(targets)
            if train:
                break
        if train:
            print 'Train evaluation finished'
        per = loss.sum() / num_examples
        return {'per': per}


def ctc_strip(sequence, blank_symbol=0):
    res = []
    for i, s in enumerate(sequence):
        if (s != blank_symbol) and (i == 0 or s != sequence[i - 1]):
            res += [s-1]

    return numpy.asarray(res)


class CTCEvaluator(object):
    def __init__(self, eol_symbol, x, x_mask, y_hat, phoneme_dict,
                 black_list=None):
        if black_list is None:
            self.black_list = []
        else:
            self.black_list = black_list
        self.x = x
        self.x_mask = x_mask
        self.eol_symbol = eol_symbol
        self.prediction_func = theano.function([x, x_mask], y_hat, on_unused_input='warn')
        self.phoneme_dict = phoneme_dict

    def evaluate(self, data_stream, train=False, file_pred=None,
                 file_targets=None):
        loss = 0.
        num_examples = 0
        iterator = data_stream.get_epoch_iterator()
        if train:
            print 'Train evaluation started'
        i = 0
        for inputs in iterator:
            inputs = dict(zip(data_stream.sources, inputs))
            x_mask_val = inputs['features_mask']
            x_val = inputs['features']
            y_val = inputs['phonemes']
            y_mask_val = inputs['phonemes_mask']
            y_hat = self.prediction_func(x_val, x_mask_val)
            y_predict = numpy.argmax(y_hat, axis=2)
            for batch in xrange(inputs['features'].shape[1]):
                y_val_cur = y_val[:sum(y_mask_val[:, batch]), batch]
                predicted = y_predict[:sum(x_mask_val[:, batch]), batch]
                predicted = ctc_strip(predicted)
                predictions = [self.phoneme_dict[phone_ind] for phone_ind
                               in predicted
                               if self.phoneme_dict[phone_ind] not in
                               self.black_list]
                targets = [self.phoneme_dict[phone_ind] for phone_ind
                           in y_val_cur
                           if self.phoneme_dict[phone_ind] not in
                           self.black_list]
                predictions = [x[0] for x in groupby(predictions)]
                targets = [x[0] for x in groupby(targets)]
                i += 1
                if file_pred:
                    file_pred.write(' '.join(predictions) + '(%d)\n' % i)
                if file_targets:
                    file_targets.write(' '.join(targets) + '(%d)\n' %i)

                loss += Evaluation.wer([predictions], [targets])
                num_examples += 1

            print '.. found sequence example:', ' '.join(predictions)
            print '.. real output was:       ', ' '.join(targets)
            if train:
                break
        if train:
            print 'Train evaluation finished'
        per = loss.sum() / num_examples
        return {'per': per}


class CTCMonitoring(SimpleExtension, MonitoringExtension):
    # TODO: create an argmax evaluator for ctc
    PREFIX_SEPARATOR = '_'

    def __init__(self, x, x_mask, y_hat, eol_symbol, data_stream, phoneme_dict,
                 black_list=None, prefix=None, train=False, **kwargs):
        super(CTCMonitoring, self).__init__(**kwargs)
        self._evaluator = CTCEvaluator(eol_symbol, x, x_mask, y_hat,
                                       phoneme_dict=phoneme_dict,
                                       black_list=black_list)
        self.data_stream = data_stream
        self.prefix = prefix
        self.train = train

    def do(self, callback_name, *args):
        """Write the values of monitored variables to the log."""
        logger.info("CTC monitoring on auxiliary data started")
        value_dict = self._evaluator.evaluate(self.data_stream, self.train)
        self.add_records(self.main_loop.log, value_dict.items())
        logger.info("CTC monitoring on auxiliary data finished")


class BeamSearchMonitoring(SimpleExtension, MonitoringExtension):
    PREFIX_SEPARATOR = '_'

    def __init__(self, samples, x, x_mask, eol_symbol, data_stream,
                 prefix=None, train=False, phoneme_dict=None, black_list=None,
                 **kwargs):
        super(BeamSearchMonitoring, self).__init__(**kwargs)
        self._evaluator = BeamSearchEvaluator(
            eol_symbol, 10, x, x_mask, samples, phoneme_dict=phoneme_dict,
            black_list=black_list)
        self.data_stream = data_stream
        self.prefix = prefix
        self.train = train

    def do(self, callback_name, *args):
        """Write the values of monitored variables to the log."""
        logger.info("Beam search monitoring on auxiliary data started")
        value_dict = self._evaluator.evaluate(self.data_stream, self.train)
        self.add_records(self.main_loop.log, value_dict.items())
        logger.info("Beam search monitoring on auxiliary data finished")


class FrameWiseEvaluator(object):
    def __init__(self, x, x_mask, y_hat, black_list=None):
        if black_list is None:
            self.black_list = []
        else:
            self.black_list = black_list
        self.x = x
        self.x_mask = x_mask
        self.prediction_func = theano.function([x, x_mask], y_hat)

    def evaluate(self, data_stream, train=False, file_pred=None,
                 file_targets=None):
        loss = 0.
        num_examples = 0
        iterator = data_stream.get_epoch_iterator()
        if train:
            print 'Train evaluation started'
        i = 0
        for inputs in iterator:
            inputs = dict(zip(data_stream.sources, inputs))
            x_mask_val = inputs['features_mask']
            x_val = inputs['features']
            y_val = inputs['triphones']
            y_mask_val = inputs['triphones_mask']
            y_hat = self.prediction_func(x_val, x_mask_val)
            y_predict = numpy.argmax(y_hat, axis=2)
            for batch in xrange(inputs['features'].shape[1]):
                y_val_cur = y_val[:sum(y_mask_val[:, batch]), batch]
                predicted = y_predict[:sum(x_mask_val[:, batch]), batch]
                predictions = [str(trihone) for triphone in predicted]
                targets = [str(triphone) for triphone in y_val_cur]
                predictions = [x[0] for x in groupby(predictions)]
                targets = [x[0] for x in groupby(targets)]
                i += 1
                if file_pred:
                    file_pred.write(' '.join(predictions) + '(%d)\n' % i)
                if file_targets:
                    file_targets.write(' '.join(targets) + '(%d)\n' %i)

                loss += (sum(numpy.not_equal(y_val_cur,
                                             predicted))).astype(
                                                     'float32') /\
                                                             len(predicted)
                num_examples += 1

            #print '.. found sequence example:', ' '.join(predictions)
            #print '.. real output was:       ', ' '.join(targets)
            if train:
                break
        if train:
            print 'Train evaluation finished'
        fer = loss / num_examples
        return {'fer': fer}

class FrameWiseMonitoring():
    PREFIX_SEPARATOR = '_'

    def __init__(self, x, x_mask, y_hat, data_stream,
                 black_list=None, prefix=None, train=False, **kwargs):
        super(FramewiseMonitoring, self).__init__(**kwargs)
        self._evaluator = FrameWiseEvaluator(x, x_mask, y_hat,
                                             black_list=black_list)
        self.data_stream = data_stream
        self.prefix = prefix
        self.train = train

    def do(self, callback_name, *args):
        """Write the values of monitored variables to the log."""
        logger.info("FrameWise monitoring on auxiliary data started")
        value_dict = self._evaluator.evaluate(self.data_stream, self.train)
        self.add_records(self.main_loop.log, value_dict.items())
        logger.info("FrameWise monitoring on auxiliary data finished")


class CTCCONVEvaluator(object):
    def __init__(self, eol_symbol, x, x_mask, y_hat, phoneme_dict,
                 black_list=None):
        if black_list is None:
            self.black_list = []
        else:
            self.black_list = black_list
        self.x = x
        self.x_mask = x_mask
        self.eol_symbol = eol_symbol
        self.prediction_func = theano.function([x, x_mask], y_hat)
        self.phoneme_dict = phoneme_dict

    def evaluate(self, data_stream, train=False, file_pred=None,
                 file_targets=None):
        loss = 0.
        num_examples = 0
        iterator = data_stream.get_epoch_iterator()
        if train:
            print 'Train evaluation started'
        i = 0
        for inputs in iterator:
            inputs = dict(zip(data_stream.sources, inputs))
            x_mask_val = inputs['features_mask']
            x_val = inputs['features']
            #transpose
            y_val = inputs['phonemes']
            y_mask_val = inputs['phonemes_mask']
            y_hat = self.prediction_func(x_val, x_mask_val)
            y_predict = numpy.argmax(y_hat, axis=2)
            for batch in xrange(inputs['features'].shape[0]):
                y_val_cur = y_val[:sum(y_mask_val[:, batch]), batch]
                predicted = y_predict[:sum(x_mask_val[:, batch]), batch]
                predicted = ctc_strip(predicted)
                predictions = [self.phoneme_dict[phone_ind] for phone_ind
                               in predicted
                               if self.phoneme_dict[phone_ind] not in
                               self.black_list]
                targets = [self.phoneme_dict[phone_ind] for phone_ind
                           in y_val_cur
                           if self.phoneme_dict[phone_ind] not in
                           self.black_list]
                predictions = [x[0] for x in groupby(predictions)]
                targets = [x[0] for x in groupby(targets)]
                i += 1
                if file_pred:
                    file_pred.write(' '.join(predictions) + '(%d)\n' % i)
                if file_targets:
                    file_targets.write(' '.join(targets) + '(%d)\n' %i)

                loss += Evaluation.wer([predictions], [targets])
                num_examples += 1

            print '.. found sequence example:', ' '.join(predictions)
            print '.. real output was:       ', ' '.join(targets)
            if train:
                break
        if train:
            print 'Train evaluation finished'
        per = loss.sum() / num_examples
        return {'per': per}


class CTCCONVMonitoring(SimpleExtension, MonitoringExtension):
    # TODO: create an argmax evaluator for ctc
    PREFIX_SEPARATOR = '_'

    def __init__(self, x, x_mask, y_hat, eol_symbol, data_stream, phoneme_dict,
                 black_list=None, prefix=None, train=False, **kwargs):
        super(CTCCONVMonitoring, self).__init__(**kwargs)
        self._evaluator = CTCCONVEvaluator(eol_symbol, x, x_mask, y_hat,
                                           phoneme_dict=phoneme_dict,
                                           black_list=black_list)
        self.data_stream = data_stream
        self.prefix = prefix
        self.train = train

    def do(self, callback_name, *args):
        """Write the values of monitored variables to the log."""
        logger.info("CTC monitoring on auxiliary data started")
        value_dict = self._evaluator.evaluate(self.data_stream, self.train)
        self.add_records(self.main_loop.log, value_dict.items())
        logger.info("CTC monitoring on auxiliary data finished")

