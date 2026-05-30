# Lesson 7 — The training loop

**Time:** ~45 minutes. Open `exercises/train.py`. This is the payoff.

Everything so far was setup. Now we make a network *learn* by repeating four steps:
forward, loss, backward, update.

## The dataset

Four tiny examples, three features each, with targets `+1` or `-1`:

```python
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]
```

The goal: get the model to output roughly the right sign for each input.

## The loop

```python
from nn import MLP
model = MLP(3, [4, 4, 1])
learning_rate = 0.05

for k in range(100):
    # 1. forward: predict every example
    ypred = [model(x) for x in xs]

    # 2. loss: mean squared error (how wrong, as one number)
    loss = sum((yo - yg) ** 2 for yg, yo in zip(ys, ypred))

    # 3. backward: clear old grads, then compute new ones
    model.zero_grad()
    loss.backward()

    # 4. update: nudge each parameter downhill
    for p in model.parameters():
        p.data += -learning_rate * p.grad

    if k % 10 == 0:
        print(k, loss.data)
```

Read step 4 carefully — it's the entire idea of learning in one line. `p.grad` says
"increasing `p` increases the loss by this much." We want the loss *down*, so we
move `p` in the **opposite** direction (the minus sign), by a small fraction (the
learning rate). Do this for all 41 parameters, repeat, and the loss falls.

## Run it

```bash
python train.py
```

You should see the loss start around `10` and plunge toward `~0`, then final
predictions land near `+1 / -1 / -1 / +1`. That tiny network just learned the
dataset using only the engine you wrote.

## Things to play with (this is where intuition forms)

- **Learning rate.** Try `0.5` — faster, maybe unstable. Try `0.001` — slow crawl.
  Too high and the loss bounces or explodes; too low and it barely moves. This
  tension never goes away, even for billion-parameter models.
- **Forget `zero_grad()`.** Comment it out. Watch training degrade as stale
  gradients accumulate across steps. This is the single most common real-world
  training bug.
- **Network size.** Try `MLP(3, [8, 8, 1])` or a single `MLP(3, [1])`. See what a
  bigger or trivially small model does.
- **Epochs.** Run for 500 steps and watch the loss flatten — diminishing returns.

## What you just did, in grown-up terms

This loop — forward, loss, backprop, gradient-descent update — is *exactly* how
GPT-scale models are trained. The differences are scale and engineering, not
concept: real models use tensors instead of scalar `Value`s (for speed on GPUs),
fancier optimizers than plain SGD (Adam), and batches of data. But the skeleton is
what's in front of you right now.

## Checkpoint
- Why the minus sign in `p.data += -learning_rate * p.grad`?
- What does the learning rate control, and what are the two failure modes if it's
  wrong?
- In one sentence: how is this loop related to how ChatGPT was trained?

Next: **Lesson 8**, sanity-checking your engine against PyTorch and looking ahead.
