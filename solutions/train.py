"""
train.py — train a tiny MLP with hand-written gradient descent.

This is the payoff of Phase 1: no PyTorch, no magic. Just the Value engine
you built, a loss, .backward(), and a manual parameter update. Watch the loss
fall and the predictions snap toward the targets.

Run:  python train.py
"""

from nn import MLP

# A tiny dataset: 4 examples, 3 features each. Targets are +1 / -1.
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]


def main():
    # 3 inputs -> 4 -> 4 -> 1 output
    model = MLP(3, [4, 4, 1])
    print(model)
    print(f"number of parameters: {len(model.parameters())}\n")

    learning_rate = 0.05
    epochs = 100

    for k in range(epochs):
        # forward pass: predictions for every example
        ypred = [model(x) for x in xs]

        # mean-squared-error loss
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

        # backward pass: zero old grads, then accumulate fresh ones
        model.zero_grad()
        loss.backward()

        # gradient descent: nudge every parameter downhill
        for p in model.parameters():
            p.data += -learning_rate * p.grad

        if k % 10 == 0 or k == epochs - 1:
            print(f"epoch {k:3d}  loss {loss.data:.6f}")

    print("\nfinal predictions vs targets:")
    for x, ygt in zip(xs, ys):
        print(f"  pred {model(x).data:+.3f}   target {ygt:+.1f}")


if __name__ == "__main__":
    main()
