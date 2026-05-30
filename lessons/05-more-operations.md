# Lesson 5 — More operations

**Time:** ~30 minutes. Still in `exercises/engine.py`.

`+` and `*` get you sums and products, but a neuron needs to *squash* its output
through a nonlinear function, and a loss needs powers and division. Each new op
follows the same recipe: compute the forward value, then write a `_backward` using
the op's local derivative.

## `__pow__` (for `x**2`, `x**-1`, etc.)

```python
def __pow__(self, other):
    assert isinstance(other, (int, float)), "only int/float powers"
    out = Value(self.data ** other, (self,), f"**{other}")
    def _backward():
        self.grad += (other * self.data ** (other - 1)) * out.grad
    out._backward = _backward
    return out
```

Local derivative is the power rule: `d(x**n)/dx = n * x**(n-1)`. With this you get
division for free (`a / b == a * b**-1`) and squaring for the loss (`x**2`).

## `tanh` — a squashing nonlinearity

```python
def tanh(self):
    t = math.tanh(self.data)
    out = Value(t, (self,), "tanh")
    def _backward():
        self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    return out
```

`tanh` maps any real number into `(-1, 1)`. Its derivative is `1 - tanh(x)**2`.
Nonlinearities are what let a stack of layers represent something more interesting
than one big linear function — without them, ten layers collapse into one.

## `relu` — the modern default

```python
def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")
    def _backward():
        self.grad += (out.data > 0) * out.grad
    out._backward = _backward
    return out
```

ReLU is `max(0, x)`: it passes positives through and zeros out negatives. Its
gradient is `1` where the output was positive and `0` otherwise. Cheap and
effective; it's what most large networks actually use.

## `exp` — for completeness (used in softmax later)

```python
def exp(self):
    out = Value(math.exp(self.data), (self,), "exp")
    def _backward():
        self.grad += out.data * out.grad   # d(e^x)/dx = e^x = out.data
    out._backward = _backward
    return out
```

## The pattern, generalized

Every differentiable op you'll ever meet fits this template:

1. Compute `out.data` from the inputs (forward).
2. Record the inputs as children.
3. In `_backward`, add `(local derivative) * out.grad` to each input's `.grad`.

That's the entire contract. Frameworks ship hundreds of ops, but each one is just
this recipe with a different line of calculus.

## Verify

```bash
python tests/test_engine.py
```

All seven tests (`add, mul, pow, tanh, relu, reused_variable, complex_expression`)
should now pass. Your engine is complete.

## Checkpoint
- Why does a network need a nonlinearity at all? What happens without one?
- `relu`'s gradient is `0` for negative inputs. What practical problem might that
  cause during training? (Hint: "dead neurons" — you'll meet this again in
  Phase 2.)

Next: **Lesson 6**, building neurons, layers, and an MLP on top of your engine.
