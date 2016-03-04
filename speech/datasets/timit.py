import os
import numpy as np
import tables
from collections import OrderedDict

from fuel.datasets import Dataset

from theano import config


class Timit(Dataset):
    """TIMIT dataset.
    
    Parameters
    ----------
    which_set : str, opt
        either 'train', 'dev' or 'test'.
    alignment : bool
        Whether return alignment.
    features : str
        The features to use. They will lead to the correct h5 file.
    """
    def __init__(self, which_set='train', alignment=False,
                 features='log_power_spectrum'):
        if features == 'log_power_spectrum':
            self.path = '/data/lisatmp3/speech/timit_alignment.h5' 
        elif features == 'fbank':
            self.path = '/data/lisatmp3/speech/timit_fbank40_for_cesar.h5'
        else:
            raise NotImplementedError
        self.features = features
        self.which_set = which_set
        print '#' * 79 
        print 'caution: 384 dev examples instead of 400'
        print '#' * 79 
        self.num_examples = {'train': 3696, 'dev': 384, 'test': 192}[which_set]
        if alignment and features == 'fbank':
            raise NotImplementedError
        if alignment:
            self.sources = ('features', 'features_shapes', 'phonemes',
                            'alignments', 'alignments_shapes')
        else:
            self.sources = ('features', 'features_shapes', 'phonemes')
        self.provides_sources = self.sources
        super(Timit, self).__init__(self.sources)
        self.open_file(self.path)

    def get_phoneme_dict(self):
        with tables.open_file(self.path) as h5file:
            phoneme_list = h5file.root._v_attrs.phones_list
            return OrderedDict(enumerate(phoneme_list))

    def get_phoneme_ind_dict(self):
        with tables.open_file(self.path) as h5file:
            phoneme_list = h5file.root._v_attrs.phones_list
            return OrderedDict(zip(phoneme_list, range(len(phoneme_list))))

    def get_normalization_factors(self):
        with tables.open_file(self.path) as h5file:
            means = h5file.root._v_attrs.means
            stds = h5file.root._v_attrs.stds
            return means, stds

    def open_file(self, path):
        self._load_in_memory()

    def _load_in_memory(self):
        """Load the data in memory and perform some transformations on them :
           1. Reshape of the features
           2. Normalization of the features
           3. Cast the features to floaX
        CAUTION: This is hardcoded for 'features', 'features_shapes' and
        'phonemes'
        
        """
        with tables.open_file(self.path) as h5file:
            means = h5file.root._v_attrs.means
            stds = h5file.root._v_attrs.stds
            node = h5file.getNode('/', self.which_set)
            nodes = [getattr(node, source) for source in self.sources]
            self.data = []
            d = []
            for i in range(len(nodes[0])):
                f = nodes[0][i].reshape(nodes[1][i])
                f -= means
                f /= stds
                d.append(f.astype(config.floatX))
            self.data.append(d)
            d = []
            for i in range(len(nodes[0])):
                d.append(nodes[2][i])
            self.data.append(d)
            paths_node = getattr(node, 'paths')
            
            #import matplotlib.pylab as plt
            #print self.data[1][5]
            #plt.figure()
            #plt.subplot(121)
            #plt.imshow(self.data[0][5].T, origin='lower', interpolation='none')
            
            # STRIPPING
            for i in range(len(self.data[0])):
                self.data[0][i], self.data[1][i] = self._strip_data_equal_length(self.data[0][i],
                                                                    self.data[1][i],
                                                                    paths_node[i][0])
            
            # DOWNSAMPLING
            #for i in range(len(self.data[0])):
            #    self.data[0][i] = self.data[0][i][::2]
            
            #print self.data[1][5]
            #plt.subplot(122)
            #plt.imshow(self.data[0][5].T, origin='lower', interpolation='none')
            #plt.show()
            
            self.data = tuple(self.data)
            self.sources = ('features', 'phonemes')
            self.provides_sources = ('features', 'phonemes')

    def get_data(self, state=None, request=None):
        if isinstance(request, slice):
            data = (self.data[0][request], self.data[1][request])
        elif isinstance(request, list):
            data = ([self.data[0][i] for i in request],
                    [self.data[1][i] for i in request])
        else:
            raise ValueError
        return data

    def _get_path(self, path):
        f = path.replace('_', '/') + '.PHN'
        if self.which_set == 'train':
            base = '/data/lisa/data/timit/raw/TIMIT/TRAIN/'
        else:
            base = '/data/lisa/data/timit/raw/TIMIT/TEST/'
        for d in ['DR1', 'DR2', 'DR3', 'DR4', 'DR5', 'DR6', 'DR7', 'DR8']:
            p = base + d + '/' + f
            if os.path.exists(p):
                return p 

    def _strip_data(self, data, labels, path):
        path = self._get_path(path)
        lines = []
        with open(path) as f:
             for line in f:
                 line = line.split()
                 line[0] = int(line[0])
                 line[1] = int(line[1])
                 lines.append(line)
        n1 = (lines[0][1] + 1) / 160 - 2
        n2 = (lines[-1][0] - 400) / 160 + 3
        labels = np.delete(labels, [1, len(labels) - 2])
        return data[n1:n2], labels
    
    
    def _strip_data_equal_length(self, data, labels, path):
        path = self._get_path(path)
        lines = []
        with open(path) as f:
             for line in f:
                 line = line.split()
                 line[0] = int(line[0])
                 line[1] = int(line[1])
                 lines.append(line)
        n1 = (lines[0][1] + 1) / 160 - 2
        stop = lines[0][1] + 160 * 100 + 400
        for i, l in enumerate(lines):
            if stop > l[0]:
                continue
            if stop < l[1]:
                break
        try:
            labels = labels[:i]
            labels = np.delete(labels, [1])
            data = data[n1:n1+100]
            return data, labels
        except IndexError:
            return data, labels
