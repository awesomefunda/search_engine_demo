# Lesson 2 — The `Value` object

**Time:** ~45 minutes. Open `exercises/engine.py`.

A plain Python `float` forgets where it came from. `2.0 + 3.0` gives `5.0` with no
memory that a `+` happened or that `2.0` and `3.0` were involved. To do
backpropagation we need that memory. So we wrap each number in a small object.

## The skeleton (already in your starter file)

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data          # the actual number
        self.grad = 0.0           # dLoss/dThis — filled in during backprop, starts at 0
        self._backward = lambda: None   # how to push grad to inputs (set per-op)
        self._prev = set(_children)     # the Values this one was built from
        self._op = _op                  # which op made it (for debugging)
```

Three ideas to internalize:
- **`data`** is the forward value.
- **`grad`** is "how much the final loss changes if this value changes". It starts
  at `0` and gets filled in later. A leaf you never touch keeps grad `0`.
- **`_prev` / `_op`** record the computation graph: every `Value` knows its parents.

## Your first task: `__add__`

When you write `a + b`, Python calls `a.__add__(b)`. We make it return a *new*
`Value` whose children are `a` and `b`:

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)  # allow a + 2
    out = Value(self.data + other.data, (self, other), "+")

    def _backward():
        self.grad  += out.grad      # TODO: fill this in
        other.grad += out.grad      # TODO: fill this in
    out._backward = _backward
    return out
```

Why `+=` and not `=`? Because a value can feed into *multiple* downstream
operations. Each contributes some gradient, and they must **add up**. (This single
detail trips up almost everyone — there's a dedicated test for it.) Why
`out.grad` on the right? Chain rule: the local derivative of `+` is `1`, so we
just pass the incoming gradient straight through (`1 * out.grad`).

## Then `__mul__`

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")

    def _backward():
        self.grad  += other.data * out.grad   # local deriv w.r.t self is `other`
        other.grad += self.data  * out.grad   # local deriv w.r.t other is `self`
    out._backward = _backward
    return out
```

This is the product rule from Lesson 1, now stored so it can run later.

## Try it (forward only for now)

```python
from engine import Value
a = Value(2.0)
b = Value(-3.0)
c = a + b
d = a * b
print(c)   # Value(data=-1.0, grad=0.0)
print(d)   # Value(data=-6.0, grad=0.0)
print(d._prev)   # {Value(a), Value(b)}  -> the graph is being recorded
```

The grads are still `0` because we haven't run backprop. The `_backward` closures
are loaded and waiting. Wiring them to fire in the right order is Lessons 3–4.

## Checkpoint
- Why must `grad` accumulate with `+=` instead of assigning with `=`?
- After `c = a + b`, what is in `c._prev`? What is `c._op`?

Next: **Lesson 3**, the computation graph and why order matters.
