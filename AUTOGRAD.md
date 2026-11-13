# Phase 1 — Backpropagation & Autograd from Scratch

> Build the engine inside every deep-learning framework, by hand, in ~150 lines.
> No PyTorch, no magic. For software engineers who want to *understand* modern AI,
> not just call its APIs.

This is **Phase 1** of a learn-modern-AI-from-scratch path modeled on Andrej
Karpathy's open-source projects. It rebuilds his
[`micrograd`](https://github.com/karpathy/micrograd): a tiny automatic
differentiation engine and a small neural network library on top of it. By the
end you'll have implemented backpropagation yourself and trained a neural net with
a hand-written gradient-descent loop.

Anyone can follow this. Clone it and go.

## Who this is for

You can program. You know functions, classes, recursion. You do **not** need any
prior machine-learning knowledge. You need a little comfort with the idea of a
derivative (slope) — Lesson 1 refreshes exactly what's required.

## What you'll be able to do at the end

- Explain precisely what `loss.backward()` does in any framework.
- Implement reverse-mode autodifferentiation from first principles.
- Build and train a small neural network with zero external libraries.
- Reason about *why* training breaks instead of guessing.

## Setup

```bash
git clone <this-repo>
cd phase1-autograd
python3 --version          # 3.8+; the core needs only the standard library
```

No dependencies are required to do the work. `pytest` (optional) makes running
tests nicer, and `torch` (optional, Lesson 8) lets you cross-check your engine
against the real thing.

```bash
pip install -r requirements.txt   # optional
```

## How to follow it

Work through the lessons in order. Each one tells you which file to edit and ends
with a checkpoint you should be able to answer.

| Lesson | Topic | You'll edit |
|---|---|---|
| [00](lessons/00-the-big-idea.md) | What "AI" means now; the mental model | — (reading) |
| [01](lessons/01-derivatives-by-hand.md) | Derivatives, numerically; the chain rule | scratch file |
| [02](lessons/02-the-value-object.md) | The `Value` object; `+` and `*` | `exercises/engine.py` |
| [03](lessons/03-the-computation-graph.md) | The graph; topological order | — (reading) |
| [04](lessons/04-backpropagation.md) | Implement `.backward()` | `exercises/engine.py` |
| [05](lessons/05-more-operations.md) | `tanh`, `relu`, `exp`, powers | `exercises/engine.py` |
| [06](lessons/06-neurons-layers-mlp.md) | `Neuron` → `Layer` → `MLP` | `exercises/nn.py` |
| [07](lessons/07-training-loop.md) | Train it; watch the loss fall | `exercises/train.py` |
| [08](lessons/08-validate-against-pytorch.md) | Cross-check vs PyTorch; what's next | — (reading) |

## Layout

```
phase1-autograd/
├── lessons/      step-by-step guide (start here, in order)
├── exercises/    starter code with TODOs — this is what YOU write
├── solutions/    reference implementation — peek only after trying
└── tests/        verify your work (gradient-checked against numerical derivatives)
```

## The workflow for each coding lesson

1. Read the lesson.
2. Fill in the `TODO`s in the indicated `exercises/` file.
3. Run the tests:
   ```bash
   python tests/test_engine.py      # or: pytest tests/
   python tests/test_nn.py
   ```
   (The tests import from `solutions/` by default so they always pass out of the
   box as a reference. To test *your* code, copy your finished
   `exercises/engine.py` and `exercises/nn.py` over the ones in `solutions/`, or
   adjust the import path at the top of each test file to point at `exercises/`.)
4. When green, run the demo:
   ```bash
   cd exercises && python train.py
   ```
5. Stuck? Compare against `solutions/`. There's no shame in it — but try first;
   the struggle is where the learning happens.

## A note on honesty with yourself

The reference solution is right there. It will be tempting to read it first. Don't.
The entire value of this phase is the moment your own `.backward()` produces a
gradient that matches calculus. Write it, get it wrong, fix it. That hour is worth
more than a week of watching tutorials.

## Credit

The design, the toy dataset, and the `micrograd` architecture are Andrej
Karpathy's, from his "Neural Networks: Zero to Hero" series. This repo is an
independent, from-scratch teaching rebuild meant to be followed hands-on.

---

**Next:** Phase 2 — `makemore`: real language models with PyTorch tensors, starting
from a bigram model and building up to text generation.
