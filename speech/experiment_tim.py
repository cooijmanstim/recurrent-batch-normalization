import argparse
import time
import os
import sys
import logging

import numpy as np

import theano
import theano.tensor as T

import ctc
from ctc_monitoring import CTCMonitoring

from blocks.bricks import Identity, Tanh, Softmax, Linear
from blocks.initialization import Constant, Orthogonal
from blocks.initialization import Identity as IdentityInit
from blocks.bricks.recurrent import SimpleRecurrent, LSTM
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                          DataStreamMonitoring)
from blocks.algorithms import (GradientDescent, StepClipping, CompositeRule,
                               Momentum, Adam, RMSProp)
from initialization import NormalizedInitialization
from blocks.model import Model
from blocks.extensions.saveload import Load, Checkpoint
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout, apply_noise
from blocks.main_loop import MainLoop
from blocks.monitoring import aggregation
from blocks.model import Model
from blocks.roles import WEIGHT, INPUT
from datasets.timit import Timit
from extensions import EarlyStopping
from utils import (construct_stream, phone_to_phoneme_dict)
import copy
import time

from bricks import TimLSTM 

floatX = theano.config.floatX
logger = logging.getLogger(__name__)
logger.setLevel('INFO')

black_list = ['<START>', '<STOP>', 'q', '<END>']


def correct_func_type(func_str):
    def func(n):
        val = eval(func_str)(n)
        return np.cast[floatX](val)
    return func


def learning_algorithm(args):
    name = args.algorithm
    learning_rate = float(args.learning_rate)
    momentum = args.momentum
    clipping_threshold = args.clipping
    if name == 'adam':
        adam = Adam(learning_rate=learning_rate)
        step_rule = adam
    elif name == 'rms_prop':
        rms_prop = RMSProp(learning_rate=learning_rate, decay_rate=0.9)
        step_rule = CompositeRule([StepClipping(1.), rms_prop])
    else:
        sgd_momentum = Momentum(learning_rate=learning_rate, momentum=momentum)
        step_rule = sgd_momentum
    return step_rule

def parse_args():
    parser = argparse.ArgumentParser(description="Selective attention "
                                                  "experiment.")
    parser.add_argument('--experiment_path', type=str,
                        default='./', help='Location for writing results')
    parser.add_argument('--label_dim', type=int,
                        default=63, help='dimension of the labels')
    parser.add_argument('--state_dim', type=int,
                        default=250, help='dimension of the state')
    parser.add_argument('--epochs', type=int,
                        default=500, help='Number of epochs')
    parser.add_argument('--seed', type=int,
                        default=123, help='Random generator seed')
    parser.add_argument('--load_path',
                        default=argparse.SUPPRESS,
                        help='File with parameter to be loaded)')
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        help='Learning rate')
    parser.add_argument('--weight_noise', type=float, default=0.,
                        help='weight_noise')
    parser.add_argument('--momentum', default=0.9,
                        type=float, help='Momentum for SGD')
    parser.add_argument('--clipping',
                        default=200,
                        type=float,
                        help='Gradient clipping norm')
    parser.add_argument('--test_cost', action='store_true',
                        default=False,
                        help='Report test set cost')
    parser.add_argument('--algorithm', choices=['rms_prop', 'adam',
                                                'sgd_momentum'],
                        default='rms_prop',
                        help='Learning algorithm to use')
    parser.add_argument('--dropout', type=bool,
                        default=False,
                        help='Use dropout in middle layers')
    parser.add_argument('--features',
                        default='fbank',
                        help='Type of features to use',
                        choices=['log_power_spectrum', 'fbank'])
    parser.add_argument('--patience', type=int,
                        default=50,
                        help='How many epochs to do before early stopping.')
    parser.add_argument('--to_watch', type=str,
                        default='dev_per',
                        help='Variable to watch for early stopping'
                             '(the smaller the better).')
    parser.add_argument('--batch_size', type=int,
                        default=24,
                        help='Size of the mini-batch')
    parser.add_argument('--batch_norm', action='store_true',
                        default=False,
                        help='Enables batch normalization')
    
    return parser.parse_args()

def train(step_rule, label_dim, state_dim, epochs,
          seed, dropout, test_cost, experiment_path, features, weight_noise,
          to_watch, patience, batch_size, batch_norm, **kwargs):

    print '.. TIMIT experiment'
    print '.. arguments:', ' '.join(sys.argv)
    t0 = time.time()


    # ------------------------------------------------------------------------
    # Streams

    rng = np.random.RandomState(seed)
    stream_args = dict(rng=rng, batch_size=batch_size)

    print '.. initializing iterators'
    train_dataset = Timit('train', features=features)
    train_stream = construct_stream(train_dataset, **stream_args)
    dev_dataset = Timit('dev', features=features)
    dev_stream = construct_stream(dev_dataset, **stream_args)
    test_dataset = Timit('test', features=features)
    test_stream = construct_stream(test_dataset, **stream_args)
    update_stream = construct_stream(train_dataset, n_batches=100,
                                     **stream_args)

    phone_dict = train_dataset.get_phoneme_dict()
    phoneme_dict = {k: phone_to_phoneme_dict[v]
                    if v in phone_to_phoneme_dict else v
                    for k, v in phone_dict.iteritems()}
    ind_to_phoneme = {v: k for k, v in phoneme_dict.iteritems()}
    eol_symbol = ind_to_phoneme['<STOP>']
 
   
    # ------------------------------------------------------------------------
    # Graph

    print '.. building model'
    x = T.tensor3('features')
    y = T.matrix('phonemes')
    input_mask = T.matrix('features_mask')
    output_mask = T.matrix('phonemes_mask')

    theano.config.compute_test_value = 'off'
    x.tag.test_value = np.random.randn(100, 24, 123).astype(floatX)
    y.tag.test_value = np.ones((30, 24), dtype=floatX)
    input_mask.tag.test_value = np.ones((100, 24), dtype=floatX)
    output_mask.tag.test_value = np.ones((30, 24), dtype=floatX)

    seq_len = 100 
    input_dim = 123 
    activation = Tanh()
    recurrent_init = IdentityInit(0.99) 

    rec1 = TimLSTM(not batch_norm, input_dim, state_dim, activation, name='LSTM')
    rec1.initialize()
    l1 = Linear(state_dim, label_dim + 1, name='out_linear',
                weights_init=Orthogonal(), biases_init=Constant(0.0))
    l1.initialize()
    o1 = rec1.apply(x)
    y_hat_o = l1.apply(o1)

    shape = y_hat_o.shape
    y_hat = Softmax().apply(y_hat_o.reshape((-1, shape[-1]))).reshape(shape)

    y_mask = output_mask
    y_hat_mask = input_mask


    # ------------------------------------------------------------------------
    # Costs and Algorithm

    ctc_cost = T.sum(ctc.cpu_ctc_th(
         y_hat_o, T.sum(y_hat_mask, axis=0),
         y + T.ones_like(y), T.sum(y_mask, axis=0)))
    batch_cost = ctc_cost.copy(name='batch_cost')

    bs = y.shape[1]
    cost_train = aggregation.mean(batch_cost, bs).copy("sequence_cost")
    cost_per_character = aggregation.mean(batch_cost,
                                          output_mask.sum()).copy(
                                                  "character_cost")
    cg_train = ComputationGraph(cost_train)

    model = Model(cost_train)
    train_cost_per_character = aggregation.mean(cost_train,
                                                output_mask.sum()).copy(
                                                        "train_character_cost")

    algorithm = GradientDescent(step_rule=step_rule, cost=cost_train,
                                parameters=cg_train.parameters,
                                on_unused_sources='warn')



    # ------------------------------------------------------------------------
    # Monitoring and extensions

    parameters = model.get_parameter_dict()
    observed_vars = [cost_train, train_cost_per_character,
                     aggregation.mean(algorithm.total_gradient_norm)]
    for name, param in parameters.iteritems():
        observed_vars.append(param.norm(2).copy(name + "_norm"))
        observed_vars.append(algorithm.gradients[param].norm(2).copy(name + "_grad_norm"))
    train_monitor = TrainingDataMonitoring(
        variables=observed_vars,
        prefix="train", after_epoch=True)

    dev_monitor = DataStreamMonitoring(
        variables=[cost_train, cost_per_character],
        data_stream=dev_stream, prefix="dev"
    )
    train_ctc_monitor = CTCMonitoring(x, input_mask, y_hat, eol_symbol, train_stream,
                                      prefix='train', every_n_epochs=1,
                                      before_training=True,
                                      phoneme_dict=phoneme_dict,
                                      black_list=black_list, train=True)
    dev_ctc_monitor = CTCMonitoring(x, input_mask, y_hat, eol_symbol, dev_stream,
                                    prefix='dev', every_n_epochs=1,
                                    phoneme_dict=phoneme_dict,
                                    black_list=black_list)

    extensions = []
    if 'load_path' in kwargs:
        extensions.append(Load(kwargs['load_path']))

    extensions.extend([FinishAfter(after_n_epochs=epochs),
                       train_monitor,
                       dev_monitor,
                       train_ctc_monitor,
                       dev_ctc_monitor])

    if test_cost:
        test_monitor = DataStreamMonitoring(
            variables=[cost_train, cost_per_character],
            data_stream=test_stream,
            prefix="test"
        )
        test_ctc_monitor = CTCMonitoring(x, input_mask, y_hat, eol_symbol, test_stream,
                                         prefix='test', every_n_epochs=1,
                                         phoneme_dict=phoneme_dict,
                                         black_list=black_list)
        extensions.append(test_monitor)
        extensions.append(test_ctc_monitor)

    #if not os.path.exists(experiment_path):
    #    os.makedirs(experiment_path)
    #best_path = os.path.join(experiment_path, 'best/')
    #if not os.path.exists(best_path):
    #    os.mkdir(best_path)
    #best_path = os.path.join(best_path, 'model.bin')
    extensions.append(EarlyStopping(to_watch, patience, '/dev/null'))
    extensions.extend([ProgressBar(), Printing()])


    # ------------------------------------------------------------------------
    # Main Loop

    main_loop = MainLoop(model=model, data_stream=train_stream,
                         algorithm=algorithm, extensions=extensions)

    print "Building time: %f" % (time.time() - t0)
   # if write_predictions:
   #     with open('predicted.txt', 'w') as f_pred:
   #         with open('targets.txt', 'w') as f_targets:
   #             evaluator = CTCEvaluator(
   #                 eol_symbol, x, input_mask, y_hat, phoneme_dict, black_list)
   #             evaluator.evaluate(dev_stream, file_pred=f_pred,
   #                                file_targets=f_targets)
   #     return
    main_loop.run()


if __name__ == '__main__':
    args = parse_args()
    step_rule = learning_algorithm(args)
    train(step_rule, **args.__dict__)
