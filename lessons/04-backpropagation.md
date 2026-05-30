# Lesson 4 — Backpropagation

**Time:** ~45 minutes. Back in `exercises/engine.py`. This is the centerpiece.

You have the pieces: each operation stored a local `_backward`, and you know the
order to run them (reverse topological). Now assemble `.backward()`.

## Implement `backward`

```python
def backward(self):
    # 1. order all nodes feeding into self
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)

    # 2. the output's gradient w.r.t. itself is 1
    self.grad = 1.0

    # 3. run each node's local backward in reverse order
    for v in reversed(topo):
        v._backward()
```

Two subtle but crucial lines:

- **`self.grad = 1.0`** — we seed the process. "How does the loss change if the
  loss changes? By exactly 1." Every other gradient flows from this seed via the
  chain rule.
- **`reversed(topo)`** — outputs before inputs, so by the time a node's
  `_backward` runs, its own `.grad` has already been fully accumulated by the nodes
  downstream of it.

## Run it

```python
from engine import Value
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
e = a * b
d = e + c
f = Value(-2.0)
L = d * f
L.backward()

print(a.grad)  # 6.0
print(b.grad)  # -4.0
print(f.grad)  # 4.0
```

Verify `a.grad = 6.0` by hand: `L = (a*b + c) * f`. So `dL/da = f * b = (-2)(-3) =
6`. The engine just computed by graph traversal what you'd derive with calculus.

## The case that breaks naive implementations

```python
a = Value(3.0)
b = a + a       # b = 2a
b.backward()
print(a.grad)   # must be 2.0
```

`a` is used twice. If `_backward` had *assigned* (`self.grad = out.grad`) instead
of *accumulating* (`self.grad += out.grad`), the second use would overwrite the
first and you'd get `1.0` — wrong. This is why every `_backward` uses `+=`. There's
a test named `test_reused_variable` guarding exactly this.

> One consequence: because grads accumulate, you must **reset them to zero** before
> each new backward pass, or gradients from the last step contaminate this one.
> That's what `zero_grad()` (coming in Lesson 6) is for.

## Verify with the test suite

```bash
cd phase1-autograd
python tests/test_engine.py     # or: pytest tests/test_engine.py
```

The tests gradient-check your engine against numerical derivatives (the wiggle
trick from Lesson 1). When `test_engine.py` is all green, your autograd engine is
provably doing calculus. That is a genuinely big milestone — pause and appreciate
it.

## Checkpoint
- Why is the seed `self.grad = 1.0` and not `0.0`?
- Explain in one sentence why `+=` (not `=`) is mandatory in every `_backward`.
- What goes wrong if you forget to zero gradients between training steps?

Next: **Lesson 5**, adding the operations a neural net needs (`tanh`, `relu`,
`exp`, `**`, division).
