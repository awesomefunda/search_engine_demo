"""
train.py — YOUR training loop. Fill in the TODOs.

Once engine.py and nn.py work, this is the satisfying part: watch a network
learn by gradient descent, written entirely by you.

Run:  python train.py
"""

from nn import MLP

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]


def main():
    model = MLP(3, [4, 4, 1])
    learning_rate = 0.05
    epochs = 100

    for k in range(epochs):
        # TODO 1 - forward pass: predict every x
        #   ypred = [model(x) for x in xs]

        # TODO 2 - loss: mean squared error between predictions and targets
        #   loss = sum((yo - yg) ** 2 for yg, yo in zip(ys, ypred))

        # TODO 3 - backward pass: zero the grads, then loss.backward()

        # TODO 4 - update: for each parameter, p.data += -learning_rate * p.grad

        # TODO 5 - print loss every 10 epochs so you can watch it fall
        pass

    # TODO 6 - print final predictions next to targets


if __name__ == "__main__":
    main()
