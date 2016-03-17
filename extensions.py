import tempfile, os.path, cPickle, zipfile, shutil, sys
from cStringIO import StringIO
from collections import OrderedDict
import numpy as np
import theano
from blocks.extensions import SimpleExtension, Printing
from blocks.serialization import secure_dump
import blocks.config

class PrintingTo(Printing):
    def __init__(self, path, **kwargs):
        super(PrintingTo, self).__init__(**kwargs)
        self.path = path
        with open(self.path, "w") as f:
            f.truncate(0)

    def do(self, *args, **kwargs):
        stdout, stringio = sys.stdout, StringIO()
        sys.stdout = stringio
        super(PrintingTo, self).do(*args, **kwargs)
        sys.stdout = stdout
        lines = stringio.getvalue().splitlines()
        with open(self.path, "a") as f:
            f.write("\n".join(lines))
            f.write("\n")

class DumpLog(SimpleExtension):
    def __init__(self, path, **kwargs):
        kwargs.setdefault("after_training", True)
        super(DumpLog, self).__init__(**kwargs)
        self.path = path

    def do(self, callback_name, *args):
        secure_dump(self.main_loop.log, self.path, use_cpickle=True)

class DumpGraph(SimpleExtension):
    def __init__(self, path, **kwargs):
        kwargs["after_batch"] = True
        super(DumpGraph, self).__init__(**kwargs)
        self.path = path

    def do(self, which_callback, *args, **kwargs):
        try:
            self.done
        except AttributeError:
            if hasattr(self.main_loop.algorithm, "_function"):
                self.done = True
                with open(self.path, "w") as f:
                    theano.printing.debugprint(self.main_loop.algorithm._function, file=f)

class DumpBest(SimpleExtension):
    """dump if the `notification_name` record is present"""
    def __init__(self, notification_name, save_path, **kwargs):
        self.notification_name = notification_name
        self.save_path = save_path
        kwargs.setdefault("after_epoch", True)
        super(DumpBest, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        if self.notification_name in self.main_loop.log.current_row:
            secure_dump(self.main_loop, self.save_path, use_cpickle=True)

from blocks.algorithms import StepRule
from blocks.roles import ALGORITHM_BUFFER, add_role
from blocks.utils import shared_floatx
from blocks.theano_expressions import l2_norm

class StepMemory(StepRule):
    def compute_steps(self, steps):
        # memorize steps for one time step
        self.last_steps = OrderedDict()
        updates = []
        for parameter, step in steps.items():
            last_step = shared_floatx(
                parameter.get_value() * 0.,
                "last_step_%s" % parameter.name)
            add_role(last_step, ALGORITHM_BUFFER)
            updates.append((last_step, step))
            self.last_steps[parameter] = last_step

        # compare last and current step directions
        self.cosine = (sum((step * self.last_steps[parameter]).sum()
                           for parameter, step in steps.items())
                       / l2_norm(steps.values())
                       / l2_norm(self.last_steps.values()))

        return steps, updates

class DumpVariables(SimpleExtension):
    def __init__(self, save_path, inputs, variables, batch, **kwargs):
        super(DumpVariables, self).__init__(**kwargs)
        self.save_path = save_path
        self.variables = variables
        self.function = theano.function(inputs, variables, on_unused_input="warn")
        self.batch = batch
        self.i = 0

    def do(self, which_callback, *args):
        values = dict((variable.name, np.asarray(value)) for variable, value in
                      zip(self.variables, self.function(**self.batch)))
        secure_dump(values, "%s_%i.pkl" % (self.save_path, self.i))
        self.i += 1
