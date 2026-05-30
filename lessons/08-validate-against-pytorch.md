# Lesson 8 — Sanity-check against PyTorch, and look ahead

**Time:** ~30 minutes. Optional install.

You don't have to take the test suite's word for it. PyTorch is the industry
autograd engine; if your gradients match its gradients on the same expression,
you've truly rebuilt the real thing.

## Install PyTorch (optional)

```bash
pip install torch
```

(If you'd rather not — the numerical gradient checks in `tests/test_engine.py`
already proved correctness. This step is for the satisfaction of seeing it line up
with the real framework.)

## Same expression, two engines

Your engine:

```python
from engine import Value
x = Value(-4.0)
z = 2 * x + 2 + x
q = z.relu() + z * x
h = (z * z).relu()
y = h + q + q * x
y.backward()
print(x.grad)
```

PyTorch:

```python
import torch
x = torch.tensor([-4.0], requires_grad=True, dtype=torch.double)
z = 2 * x + 2 + x
q = z.relu() + z * x
h = (z * z).relu()
y = h + q + q * x
y.backward()
print(x.grad.item())
```

Both print the same number. Same forward value, same gradient. Notice how similar
the code reads — that's not a coincidence. The `Value` API was modeled on PyTorch's
tensor API on purpose, so the mental model you built here transfers directly.

The only real difference: PyTorch's `tensor` holds an *array* of numbers and runs
operations on all of them at once (vectorized, on a GPU). Your `Value` holds one
scalar. Same calculus, different data layout for speed. When you see
`loss.backward()` in any real codebase from now on, you know exactly what it does:
topological sort, seed the output grad to 1, run each op's local backward in
reverse, accumulating into `.grad`.

## What you've actually accomplished

You built, from nothing:
- a working reverse-mode automatic differentiation engine,
- a neural network library on top of it,
- and a training loop that learns by gradient descent —

with no libraries, in ~150 lines. This is the irreducible core of every system in
modern AI, including the model you're reading this from.

## Self-test before moving on

You're ready for Phase 2 if you can, without looking:
1. Explain what `.grad` means and why `_backward` uses `+=`.
2. Describe the four steps of a training loop and why the update has a minus sign.
3. Say what a topological sort is for in `.backward()`.
4. Explain why nonlinearities (tanh/relu) are necessary.

If any feel shaky, re-skim that lesson and re-run the relevant code — understanding
here pays compounding interest in every later phase.

## Where this goes next (Phase 2 preview)

The big limitation you just felt: scalar `Value`s are slow. Phase 2 (`makemore`)
switches to PyTorch tensors and builds real language models — starting from a
bigram character model and working up to a multi-layer perceptron that generates
text. You'll meet embeddings, proper training/validation splits, and the
diagnostic skills (initialization, normalization, learning-rate finding) that
separate people who *use* models from people who can *fix* them.

Everything there rests on what you just built. Nice work.

---

## Further reading
- Andrej Karpathy, "The spelled-out intro to neural networks and backpropagation:
  building micrograd" (the video this phase is modeled on).
- The original `micrograd` repository — compare your engine to his; they're
  deliberately close.
