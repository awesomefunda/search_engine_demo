# Lesson 3 — The computation graph

**Time:** ~30 minutes. Conceptual, with a small experiment.

Every time you combine `Value`s, you extend a graph. Nodes are `Value`s; edges
point from a result back to the inputs that produced it (that's what `_prev`
stores). Consider:

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
e = a * b      # e depends on a, b
d = e + c      # d depends on e, c
f = Value(-2.0)
L = d * f      # L depends on d, f
```

The graph (arrows = "is built from"):

```
a ─┐
   ├─(*)→ e ─┐
b ─┘         ├─(+)→ d ─┐
c ───────────┘         ├─(*)→ L
f ─────────────────────┘
```

`L` is the final output (think: the loss). To compute every gradient we want
`dL/dx` for each node `x`, all the way back to the leaves `a, b, c, f`.

## Why order matters

To compute `a.grad` correctly, we first need `e.grad`. To know `e.grad`, we first
need `d.grad`. To know `d.grad`, we need `L.grad`. In other words: **a node's
gradient can only be computed once all the nodes that depend on it are done.**

So we must process nodes in reverse order of the graph — outputs first, leaves
last. The standard way to get a valid order is a **topological sort**: a linear
ordering where every node comes *after* all of its children (inputs). Then we walk
it backward.

## Topological sort by depth-first search

```python
topo = []
visited = set()

def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:   # visit inputs first
            build_topo(child)
        topo.append(v)          # add self only after all inputs are added
```

After `build_topo(L)`, `topo` lists nodes leaves-first, output-last. Reverse it and
you get output-first, which is exactly the order to run the `_backward` closures.

> Mental check: a child is always appended before its parent, because we recurse
> into children *before* appending self. So in `topo`, inputs come before the
> results that use them. Reversed, results come before their inputs — correct for
> backprop.

## Visualize it yourself

Add a quick printout to see the recorded graph for the expression above:

```python
def show(v, indent=0):
    print(" " * indent + f"{v._op or 'leaf'}  data={v.data}")
    for child in v._prev:
        show(child, indent + 2)

show(L)
```

You'll see `L` at the top, branching down into `d` and `f`, then `e` and `c`, then
`a` and `b`. That tree is precisely what backprop will traverse — in reverse.

## Checkpoint
- Why can't we compute `a.grad` before `e.grad`?
- In the topo list, does a node appear before or after its `_prev` children?
- What ordering do we actually iterate during backprop — the topo list, or its
  reverse?

Next: **Lesson 4**, where you implement `.backward()` and finally get real
gradients.
