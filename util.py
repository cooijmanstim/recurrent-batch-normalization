from collections import OrderedDict
import numpy as np, theano, theano.tensor as T
import itertools as it
from picklable_itertools.extras import equizip

# scan with arguments in dicts rather than lists
def scan(fn,
         sequences=None,
         outputs_info=None,
         non_sequences=None,
         **scan_kwargs):
    # we don't care about the order, as long as it's consistent
    sequences = OrderedDict(sequences or [])
    outputs_info = OrderedDict(outputs_info or [])
    non_sequences = OrderedDict(non_sequences or [])

    # make sure names are unique
    assert not (set(sequences) & set(outputs_info) & set(non_sequences))

    def listified_fn(*input_list):
        input_dict = OrderedDict()
        input_it = iter(input_list)
        input_dict.update(equizip(sequences.keys(),
                                  it.islice(input_it, len(sequences))))
        for name, info in outputs_info.items():
            if info is None:
                continue # no inputs
            elif isinstance(info, (dict, OrderedDict)):
                ntaps = len(info.get("taps", [-1]))
            else:
                # assume some kind of tensor variable or numpy array
                ntaps = 1
            taps = [next(input_it) for _ in range(ntaps)]
            input_dict[name] = taps if ntaps > 1 else taps[0]
        input_dict.update(equizip(non_sequences.keys(),
                                  it.islice(input_it, len(non_sequences))))

        # input_list should be exactly empty here
        try:
            next(input_it)
        except StopIteration:
            pass
        else:
            assert False

        output_dict = fn(**input_dict)
        output_list = [output_dict[output_name].copy(name=output_name)
                       for output_name in outputs_info.keys()]
        return output_list

    outputs, updates = theano.scan(
        listified_fn,
        sequences.values(),
        outputs_info.values(),
        non_sequences.values(),
        **scan_kwargs)
    outputs = OrderedDict(equizip(outputs_info.keys(), outputs))
    return outputs, updates
