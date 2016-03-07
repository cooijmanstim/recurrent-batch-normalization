import numpy

import theano
from blocks.search import BeamSearch


floatX = theano.config.floatX


class BeamSearchLM(BeamSearch):
    """Works with word level output
    """
    def __init__(self, lm, beta, ind_to_word, *args, **kwargs):
        self.lm = lm
        self.beta = beta
        self.ind_to_word = ind_to_word
        self.all_outputs = None
        super(BeamSearchLM, self).__init__(*args, **kwargs)

    def compute_initial_states(self, contexts):
        states = super(BeamSearchLM, self).compute_initial_states(contexts)
        # We have to aggregate the output text here
        self.all_outputs = numpy.array([states['outputs']])
        return states

    def lm_logprobs(self):
        logprobs = []
        for i in xrange(self.all_outputs.shape[1]):
            sequence = self.all_outputs[:, i]
            word_sequence = [self.ind_to_word[ind] for ind in sequence]
            logprobs.append(self.lm.prob(word_sequence[-1],
                                         word_sequence[:-1]))
        return numpy.array(logprobs)

    def compute_logprobs(self, contexts, states):
        am_logprobs = super(BeamSearchLM, self).compute_logprobs(
            contexts, states)
        self.all_outputs = numpy.vstack([self.all_outputs,
                                         states['outputs'][None, :]])
        return am_logprobs - self.beta * self.lm_logprobs()


class Evaluation(object):
    @classmethod
    def levenshtein(cls, predicted_seq, target_seq, predicted_mask=None,
                    target_mask=None, eol_symbol=-1):
        """
        Informally, the Levenshtein distance between two
        sequences is the minimum number of symbol edits
        (i.e. insertions, deletions or substitutions) required to
        change one word into the other. (From Wikipedia)
        """
        if predicted_mask is None:
            plen, tlen = len(predicted_seq), len(target_seq)
        else:
            assert len(target_mask) == len(target_seq)
            assert len(predicted_mask) == len(predicted_seq)
            plen, tlen = int(sum(predicted_mask)), int(sum(target_mask))

        dist = [[0 for i in range(tlen+1)] for x in range(plen+1)]
        #dist = np.zeros((plen, tlen), dype=floatX)
        for i in xrange(plen+1):
            dist[i][0] = i
        for j in xrange(tlen+1):
            dist[0][j] = j

        for i in xrange(plen):
            for j in xrange(tlen):
                if predicted_seq[i] != target_seq[j]:
                    cost = 1  
                else:
                    cost = 0
                dist[i+1][j+1] = min(dist[i][j+1] + 1,   # deletion
                                     dist[i+1][j] + 1,   # insertion
                                     dist[i][j] + cost)   # substitution

        return dist[-1][-1]

    @classmethod
    def wer(cls, predicted_seq, target_seq, predicted_mask=None,
            target_mask=None,
            eol_symbol=-1):
        """
        Word Error Rate is 'levenshtein distance' devided by
        the number of elements in the target sequence.

        Input may also be batches of data
        """
        if predicted_mask is None:
            error_rate = []
            for (single_predicted_seq, single_target_seq) in zip(predicted_seq,
                                                                 target_seq):

                l_dist = cls.levenshtein(predicted_seq=single_predicted_seq,
                                         target_seq=single_target_seq,
                                         eol_symbol=eol_symbol)
                error_rate += [l_dist / float(len(single_target_seq))]
        else:
            error_rate = []
            # iteration over columns
            for (single_predicted_seq, single_p_mask,
                 single_target_seq, single_t_mask) in zip(predicted_seq.T,
                                                          predicted_mask.T,
                                                          target_seq.T,
                                                          target_mask.T):

                l_dist = cls.levenshtein(predicted_seq=single_predicted_seq,
                                         target_seq=single_target_seq,
                                         predicted_mask=single_p_mask,
                                         target_mask=single_t_mask,
                                         eol_symbol=eol_symbol)
                error_rate += [l_dist / sum(single_t_mask)]    
        # returns an array for every single example in the batch
        return numpy.array(error_rate)
