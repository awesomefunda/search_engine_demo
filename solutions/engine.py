"""
engine.py — a tiny scalar-valued autograd engine.

This is the reference solution. Each `Value` wraps a single number, remembers
how it was produced (its "children" and the operation), and knows how to push
gradients backward to those children via the chain rule.

If you understand every line in this file, you understand the core idea behind
PyTorch, TensorFlow, JAX, and every modern deep-learning framework.
"""

import math


class Value:
    """Stores a single scalar value and its gradient."""

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        # internal: the function that propagates gradient from this node to its inputs
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for debugging/printing

    # ---- core operations: each returns a new Value AND wires up its local gradient ----

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(out)/d(self) = 1, d(out)/d(other) = 1  -> just pass the gradient through
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(out)/d(self) = other,  d(out)/d(other) = self
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            # d(x**n)/dx = n * x**(n-1)
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # d(tanh(x))/dx = 1 - tanh(x)**2
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            # d(exp(x))/dx = exp(x) = out.data
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    # ---- the heart of it: backpropagation over the whole graph ----

    def backward(self):
        # 1. build a topological order of all nodes feeding into self
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # 2. seed the output gradient, then apply chain rule in reverse order
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    # ---- convenience operators so we can write natural math ----

    def __neg__(self):
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
