from collections import defaultdict
import matplotlib.pylab as plt
import sys
import os

plt.rcParams.update({'font.size': 16})
plt.close('all')


def parse_log(file_):
    data = defaultdict(list)
    with open(file_) as f:
        for line in f:
            line = line.strip()
            line = line.split()
            if len(line) == 2:
                try:
                    data[line[0][:-1]].append(float(line[1]))
                except ValueError:
                    continue
    return data


def plot(logs):
    plt.figure()
    plt.title('TIMIT')
    colors = ['b', 'r', 'g', 'k', 'y', 'c', 'm', 'b', 'r', 'g', 'k', 'y', 'c', 'm']
    for i, log in enumerate(logs):
        name = os.path.basename(log)
        data = parse_log(log)
        plt.plot(data['train_per'][1:-1], c=colors[i], ls=':', lw=2)
        plt.plot(data['dev_per'][:-1], c=colors[i], lw=2, label=name)
    plt.grid()
    plt.ylim([0, 1])
    #plt.xlim([0, 100])
    plt.xlabel('Epochs'); plt.ylabel('Phoneme Error Rate (PER)')
    plt.legend()
    plt.show()
    plt.close('all')


logs = [#'results/original_baseline_24.txt',
        #'results/original_bn_24.txt',
        #'results/rms_identity_baseline_24.txt',
        #'results/LSTM1L_rms_identity_baseline_24.txt',
        #'results/LSTM1L_rms_identity_noise_bn_24.txt',
        'results/tim_baseline.txt',
        'results/tim_bn.txt',
        'results/tim_baseline_48.txt',
        'results/tim_bn_48.txt',
        'results/tim_baseline_96.txt',
        'results/tim_bn_96.txt'
       ]
plot(logs)
