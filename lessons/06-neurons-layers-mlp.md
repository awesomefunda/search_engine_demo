# Lesson 6 — Neurons, Layers, and the MLP

**Time:** ~45 minutes. Open `exercises/nn.py`.

Your engine handles arbitrary scalar math with gradients. A neural network is just
a *particular, repeated* arrangement of that math. We'll build it in three layers
of abstraction — the same hierarchy PyTorch uses (`nn.Module`).

## A `Module` base class

```python
class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
    def parameters(self):
        return []
```

`parameters()` returns every trainable `Value`. `zero_grad()` resets their grads
before each backward pass — remember from Lesson 4 why this is mandatory (grads
accumulate).

## A `Neuron`

A neuron computes `tanh(w · x + b)`: a weighted sum of its inputs plus a bias,
squashed.

```python
class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act
    def parameters(self):
        return self.w + [self.b]
```

The weights start random (symmetry-breaking — if every weight were identical, every
neuron would learn the same thing). The bias starts at `0`. Notice `act` is a
`Value` built from multiplies and adds, so it already carries a full backward
graph. You wrote that machinery; here you're just *using* it.

## A `Layer`

A layer is a row of independent neurons all reading the same input:

```python
class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
```

`nin` inputs in, `nout` outputs out.

## An `MLP` (multi-layer perceptron)

Stack layers; the output of one feeds the next. Hidden layers are nonlinear; the
final layer is usually linear so the output isn't squashed into `(-1, 1)`
artificially.

```python
class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i+1], nonlin=(i != len(nouts) - 1))
            for i in range(len(nouts))
        ]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

## Try it

```python
from nn import MLP
model = MLP(3, [4, 4, 1])   # 3 inputs -> 4 -> 4 -> 1 output
x = [2.0, 3.0, -1.0]
print(model(x))                  # a single Value
print(len(model.parameters()))   # 41
```

A forward pass through `model(x)` silently builds a computation graph with dozens
of nodes. Calling `.backward()` on the result would fill in `.grad` for all 41
parameters at once — that's the power of what you built in Phase 1.

## Verify

```bash
python tests/test_nn.py
```

## Checkpoint
- Why initialize weights randomly instead of all-zero?
- An `MLP(3, [4, 4, 1])` has how many parameters, and why 41? (Work it out:
  `4*(3+1) + 4*(4+1) + 1*(4+1)`.)
- Where, exactly, is the autograd graph being constructed when you call `model(x)`?

Next: **Lesson 7**, the training loop — making it actually learn.
