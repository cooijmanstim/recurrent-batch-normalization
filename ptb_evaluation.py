# import this before theano
from nowwithpopulationstatistics import get_population_outputs

import sys, pprint
import theano, numpy as np
import fuel
from penntreebank import PTB
from blocks.serialization import load

def get_stream(which_set, batch_size, length, num_examples=None):
    dataset = PTB(which_set=which_set, length=length)
    if num_examples is None or num_examples > dataset.num_examples:
        num_examples = dataset.num_examples
    stream = fuel.streams.DataStream.default_stream(
        dataset,
        iteration_scheme=fuel.schemes.ShuffledScheme(num_examples, batch_size))
    return stream

main_loop = load(sys.argv[1])
valid_monitor = next(extension for extension in main_loop.extensions
                     if getattr(extension, "prefix", None) == "valid")

inputs = valid_monitor._evaluator.unique_inputs
outputs = dict()
outputs["batch"] = valid_monitor._evaluator.theano_variables
outputs["population"] = get_population_outputs(
    inputs, outputs["batch"],
    main_loop.data_stream.get_epoch_iterator(as_dict=True))
functions = dict((key, theano.function(inputs, outputs[key]))
                 for key in outputs)

def evaluate(function, outputs, batches):
    values = {}
    for i, batch in enumerate(batches):
        for output, value in zip(outputs, function(**batch)):
            if output not in values:
                values[output] = np.zeros_like(value)
            values[output] *= i / float(i + 1)
            values[output] += 1 / float(i + 1) * value
        sys.stdout.write(".")
        sys.stdout.flush()
    sys.stdout.write("\n")
    return values

lengths = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
valuess = []
for length in lengths:
    evaluation_stream = get_stream("test", 100, length)
    values = dict((key, evaluate(functions[key], [output.name for output in outputs[key]],
                                 evaluation_stream.get_epoch_iterator(as_dict=True)))
                  for key in outputs)
    print "length %i:" % length
    for key in outputs:
        print key, values[key]
    valuess.append(values)

import matplotlib.pyplot as plt
for output in "cross_entropy error_rate".split():
    plt.figure()
    for key in functions:
        plt.plot(lengths, np.asarray([values[key][output] for values in valuess]),
                 label=key)
    plt.title("Penn Treebank batch statistics vs time-averaged population statistics")
    plt.ylabel(output)
    plt.xlabel("sequence length")
    plt.legend()
plt.show()

import pdb; pdb.set_trace()

# proper evaluation on one long sequence:
test_stream = get_stream("test", 1, 446184)
values = evaluate(functions["population"], outputs["population"], test_stream.get_epoch_iterator(as_dict=True))
print values

import pdb; pdb.set_trace()
