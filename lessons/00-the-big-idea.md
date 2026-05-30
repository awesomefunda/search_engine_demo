# Lesson 0 — The Big Idea

**Time:** ~30 minutes of reading. No code yet.

You're a software engineer. You already know how to write a function: you specify
the rules, the computer follows them. Modern AI flips this. Instead of writing the
rules, you write a function with millions of unknown *parameters*, then let an
algorithm tune those parameters until the function does what you want — judged
only by examples.

That tuning algorithm is **gradient descent**, and the thing that makes it possible
is **backpropagation**. Phase 1 is entirely about building both, by hand, for the
simplest possible case (single numbers), so they stop being magic.

## The mental model

A neural network is just a big mathematical expression. You feed numbers in
(the input), they flow through a chain of multiplies, adds, and squashing
functions, and a number comes out (the prediction). Somewhere you compare that
prediction to the right answer and get a single number called the **loss** — how
wrong you were. Big loss = bad.

Training is one question asked over and over:

> "If I nudge each parameter a tiny bit, which direction makes the loss go down?"

The answer for one parameter is its **gradient** — the derivative of the loss with
respect to that parameter. Backpropagation is a clever way to compute the gradient
of *every* parameter in one backward sweep through the expression, no matter how
big. Then you take a small step in the downhill direction. Repeat thousands of
times and the function learns.

## Why build it from scratch

You will never write your own autograd engine at work — PyTorch does it for you.
But after this phase, `loss.backward()` will never be mysterious again, and you'll
be able to reason about *why* training breaks (vanishing gradients, dead neurons,
exploding values) instead of guessing. This is the foundation every later phase
stands on.

## What you'll build in Phase 1

1. A `Value` object that wraps a number and tracks how it was computed.
2. Backpropagation over a graph of `Value`s.
3. A `Neuron`, `Layer`, and `MLP` (multi-layer perceptron) on top of it.
4. A training loop that makes a tiny network learn a dataset — with zero
   external libraries.

This is a faithful, from-scratch rebuild of Andrej Karpathy's `micrograd`. About
150 lines of code teach you the engine inside every deep-learning framework.

## One prerequisite refresher: the derivative

The derivative of `f` at `x` answers: "if I increase `x` by a tiny amount, how much
does `f(x)` change, and in which direction?" Formally it's the slope:

```
f'(x) = lim (h -> 0)  ( f(x + h) - f(x) ) / h
```

You only need three facts going forward:
- **Sum rule:** derivative of `a + b` w.r.t. `a` is `1`.
- **Product rule:** derivative of `a * b` w.r.t. `a` is `b`.
- **Chain rule:** if `y` depends on `u` and `u` depends on `x`, then
  `dy/dx = dy/du * du/dx`. (This is the whole game. Backprop is just the chain
  rule applied across a graph.)

Next: **Lesson 1**, where you'll compute a gradient by hand so the code later
feels obvious.
