"""
test_nn.py — verify the neural net library works.

Run either:
    pytest tests/
    python tests/test_nn.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "solutions"))

from engine import Value
from nn import Neuron, Layer, MLP


def test_neuron_shapes():
    n = Neuron(3)
    out = n([1.0, 2.0, 3.0])
    assert isinstance(out, Value)
    # 3 weights + 1 bias
    assert len(n.parameters()) == 4


def test_layer_shapes():
    layer = Layer(3, 4)
    out = layer([1.0, 2.0, 3.0])
    assert isinstance(out, list) and len(out) == 4
    assert len(layer.parameters()) == 4 * (3 + 1)


def test_mlp_param_count():
    model = MLP(3, [4, 4, 1])
    # layer1: 4*(3+1)=16, layer2: 4*(4+1)=20, layer3: 1*(4+1)=5  -> 41
    assert len(model.parameters()) == 41


def test_mlp_can_learn():
    """A short training run should drive the loss down substantially."""
    import random
    random.seed(1337)

    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]
    model = MLP(3, [4, 4, 1])

    def total_loss():
        ypred = [model(x) for x in xs]
        return sum((yo - yg) ** 2 for yg, yo in zip(ys, ypred))

    first = total_loss().data
    for _ in range(50):
        loss = total_loss()
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data += -0.05 * p.grad
    last = total_loss().data

    assert last < first * 0.1, f"loss barely moved: {first} -> {last}"


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
