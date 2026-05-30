"""
test_engine.py — verify your autograd engine is correct.

Strategy: "gradient checking". For each operation we compute the gradient two
ways and check they agree:
  1. analytically, via your .backward()
  2. numerically, by nudging an input a tiny bit and seeing how the output moves
     (the definition of a derivative: (f(x+h) - f(x-h)) / 2h)

If these match, your backward pass is implementing calculus correctly.

Run either:
    pytest tests/
    python tests/test_engine.py        (no pytest needed)
"""

import sys
import os

# allow importing the engine whether you point this at solutions/ or exercises/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "solutions"))

from engine import Value

H = 1e-6
TOL = 1e-4


def numeric_grad(f, x, h=H):
    """Numerically estimate df/dx at the scalar point x."""
    return (f(x + h) - f(x - h)) / (2 * h)


def test_add():
    a = Value(2.0)
    b = Value(-3.0)
    c = a + b
    c.backward()
    assert abs(a.grad - 1.0) < TOL
    assert abs(b.grad - 1.0) < TOL


def test_mul():
    a = Value(2.0)
    b = Value(-3.0)
    c = a * b
    c.backward()
    assert abs(a.grad - (-3.0)) < TOL  # dc/da = b
    assert abs(b.grad - 2.0) < TOL     # dc/db = a


def test_pow():
    a = Value(3.0)
    c = a ** 2
    c.backward()
    assert abs(a.grad - 6.0) < TOL  # d(a^2)/da = 2a = 6


def test_tanh():
    a = Value(0.7)
    c = a.tanh()
    c.backward()
    expected = numeric_grad(lambda x: __import__("math").tanh(x), 0.7)
    assert abs(a.grad - expected) < TOL


def test_relu():
    pos = Value(2.0)
    pos.relu().backward()
    assert abs(pos.grad - 1.0) < TOL
    neg = Value(-2.0)
    neg.relu().backward()
    assert abs(neg.grad - 0.0) < TOL


def test_reused_variable():
    """The tricky case: a node used more than once must ACCUMULATE gradient."""
    a = Value(3.0)
    b = a + a  # b = 2a, so db/da should be 2, not 1
    b.backward()
    assert abs(a.grad - 2.0) < TOL


def test_complex_expression():
    """Full gradient check on a messy expression of two variables."""
    def forward(av, bv):
        a, b = Value(av), Value(bv)
        d = a * b + b ** 3
        d = d.tanh() + a * a
        e = (d + a) * a
        return a, b, e

    a, b, e = forward(2.0, -1.5)
    e.backward()

    ga = numeric_grad(lambda x: forward(x, -1.5)[2].data, 2.0)
    gb = numeric_grad(lambda x: forward(2.0, x)[2].data, -1.5)
    assert abs(a.grad - ga) < 1e-3, f"grad a: {a.grad} vs {ga}"
    assert abs(b.grad - gb) < 1e-3, f"grad b: {b.grad} vs {gb}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as ex:
            print(f"FAIL  {t.__name__}: {ex}")
    print(f"\n{passed}/{len(tests)} tests passed")
