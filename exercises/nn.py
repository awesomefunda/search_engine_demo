"""
nn.py — YOUR neural net library. Fill in the TODOs.

Build it on top of the engine.py you just wrote. By the end you'll have
Neuron -> Layer -> MLP, the same hierarchy PyTorch uses.

Import note: this imports `from engine import Value`. Run your code from inside
this folder, or copy your finished engine.py next to this file.
"""

import random
from engine import Value


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        # TODO: create `nin` weights as Values with small random data in [-1, 1],
        #       and one bias Value initialized to 0.0
        #   self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        #   self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        # TODO: compute the weighted sum of inputs plus bias:
        #   act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # then apply tanh if self.nonlin else return act directly
        pass

    def parameters(self):
        # TODO: return all weights plus the bias as one list
        pass


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        # TODO: create `nout` neurons, each taking `nin` inputs
        pass

    def __call__(self, x):
        # TODO: run every neuron on x. Return a single Value if there's one
        #       neuron, otherwise the list.
        pass

    def parameters(self):
        # TODO: flatten the parameters of all neurons into one list
        pass


class MLP(Module):
    def __init__(self, nin, nouts):
        # nouts is a list of layer sizes, e.g. [4, 4, 1].
        # TODO: build layers so layer i maps sizes sz[i] -> sz[i+1], where
        #       sz = [nin] + nouts. Make the LAST layer linear (nonlin=False)
        #       and all earlier layers nonlinear.
        sz = [nin] + nouts
        pass

    def __call__(self, x):
        # TODO: pass x through each layer in turn, returning the final output
        pass

    def parameters(self):
        # TODO: flatten parameters of all layers
        pass
