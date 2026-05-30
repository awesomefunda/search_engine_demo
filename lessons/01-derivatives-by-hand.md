# Lesson 1 — Derivatives, the engineer's way

**Time:** ~30 minutes. A little code (a scratch file), no engine yet.

Before we automate derivatives, let's get an intuitive, numerical feel for them.
Open a Python REPL or scratch file.

## Derivative = "wiggle the input, watch the output"

You don't need symbolic calculus to *see* a gradient. Just nudge an input by a
tiny `h` and measure the change:

```python
def f(x):
    return 3 * x**2 - 4 * x + 5

h = 0.0001
x = 3.0
slope = (f(x + h) - f(x)) / h
print(slope)   # ~14.0
```

That number is the slope of `f` at `x = 3`. Positive slope means "increasing x
increases f". If `f` were a loss, you'd move `x` the *other* way to decrease it.

This trick — the **numerical gradient** — is slow and imprecise, but it's gold for
*checking* a real implementation. Our test suite uses it to verify the engine you
build is doing real calculus. (You can't ship it, though: a network with a million
parameters would need a million re-evaluations per step.)

## Now with multiple inputs

Real expressions have many inputs. The gradient is just the slope with respect to
*each* one, holding the others fixed:

```python
def f(a, b, c):
    return a * b + c

h = 0.0001
a, b, c = 2.0, -3.0, 10.0
d1 = f(a, b, c)
print((f(a + h, b, c) - d1) / h)   # ~ -3.0  (= b)
print((f(a, b + h, c) - d1) / h)   # ~  2.0  (= a)
print((f(a, b, c + h) - d1) / h)   # ~  1.0
```

Notice the gradient of `a*b + c` w.r.t. `a` came out to `b`, w.r.t. `b` came out
to `a`, and w.r.t. `c` came out to `1`. That's the product rule and sum rule,
confirmed numerically. Keep this in mind — you're about to hard-code exactly these
rules into each operation.

## The chain rule, concretely

Now compose functions:

```python
# y = (a * b) then squashed by tanh
import math
def f(a, b):
    return math.tanh(a * b)
```

To get `dy/da` by hand: let `u = a*b`. Then `dy/da = dy/du * du/da`.
- `dy/du = 1 - tanh(u)**2`  (the derivative of tanh)
- `du/da = b`

So `dy/da = (1 - tanh(a*b)**2) * b`. **Backpropagation is nothing more than this
chain rule, applied one operation at a time, walking backward through a whole
expression.** Each operation only needs to know its *own* local derivative; the
chain rule stitches them together.

## Checkpoint

You should now be able to answer, without running code:
- What is the gradient of `q = a + b` w.r.t. `a`? (Answer: 1)
- What is the gradient of `q = a * b` w.r.t. `b`? (Answer: a)
- If `L = q * c` and `q = a + b`, what is `dL/da`?
  (Answer: `dL/dq * dq/da = c * 1 = c`.)

That last one — pushing the gradient from `L` back through `q` to `a` — is the
exact move you'll automate next.

Next: **Lesson 2**, where we wrap numbers in a `Value` object that remembers how
it was built.
