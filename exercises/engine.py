"""
engine.py — YOUR autograd engine. Fill in the TODOs.

Goal: make `python ../tests/test_engine.py` pass (point it at this folder, see
the note at the bottom of the test file, or just copy this file's folder onto
the path). Work top to bottom. The lessons in ../lessons/ walk you through each
piece. Peek at ../solutions/engine.py only after you've tried.

The whole trick: every operation does two things.
  1. compute the forward result (the easy part)
  2. define a local `_backward()` that says how gradient flows to the inputs
     (the chain rule, applied to just this one operation)
"""

import math


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # TODO: addition passes gradient straight through to both inputs.
            #   self.grad  += ???
            #   other.grad += ???
            pass

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # TODO: product rule. d(a*b)/da = b, d(a*b)/db = a.
            #   self.grad  += ??? * out.grad
            #   other.grad += ??? * out.grad
            pass

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float powers"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            # TODO: power rule. d(x**n)/dx = n * x**(n-1).
            #   self.grad += ??? * out.grad
            pass

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            # TODO: gradient passes through only where the output was positive.
            #   self.grad += (out.data > 0) * out.grad
            pass

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # TODO: d(tanh(x))/dx = 1 - tanh(x)**2   (note: t is tanh(x))
            #   self.grad += ??? * out.grad
            pass

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            # TODO: d(exp(x))/dx = exp(x), which is exactly out.data
            #   self.grad += ??? * out.grad
            pass

        out._backward = _backward
        return out

    def backward(self):
        # TODO (two parts):
        # 1. Build `topo`: a list of all nodes feeding into self, ordered so that
        #    every node appears AFTER all of its children. Use the recursive
        #    depth-first helper sketched below.
        # 2. Set self.grad = 1.0, then call v._backward() for v in reversed(topo).
        topo = []
        visited = set()

        def build_topo(v):
            # TODO: if v not visited: mark visited, recurse into v._prev, append v
            pass

        build_topo(self)
        # TODO: seed the gradient and walk backwards
        pass

    # ---- these are provided for you (they just lean on the ops above) ----
    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
