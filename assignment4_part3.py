"""Assignment 4 - Part 3: gradient mechanics in PyTorch (Questions 6-9).

Run from the project root:
    uv run python assignment4_part3.py
"""
import torch

print("=" * 60)
print("Question 6: Basic gradient calculations")
print("=" * 60)

# 6a) y = x^2 + 3x  ->  dy/dx = 2x + 3 = 2(2) + 3 = 7
x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3 * x
y.backward()
print("6a) x.grad =", x.grad)  # tensor([7.])

# 6b) requires_grad=False -> no graph is built, so backward() fails
x = torch.tensor([2.0], requires_grad=False)
y = x**2 + 3 * x
try:
    y.backward()
except RuntimeError as e:
    print("6b) RuntimeError:", e)

# 6c) The default is requires_grad=False
x = torch.tensor([2.0])
print("6c) default requires_grad =", x.requires_grad)  # False

print()
print("=" * 60)
print("Question 7: Introduce weights")
print("=" * 60)

# 7a) w was created without requires_grad -> PyTorch does not track it -> w.grad is None
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([1.0, 3.0])
y = w[0] * x**2 + w[1] * x
y.backward()
print("7a) x.grad =", x.grad)  # 2*w0*x + w1 = 2*1*2 + 3 = 7
print("7a) w.grad =", w.grad)  # None

# 7b) Track gradients for w:  dy/dw0 = x^2 = 4,  dy/dw1 = x = 2
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([1.0, 3.0], requires_grad=True)
y = w[0] * x**2 + w[1] * x
y.backward()
print("7b) x.grad =", x.grad)  # tensor([7.])
print("7b) w.grad =", w.grad)  # tensor([4., 2.])

# 7c) Same as 6c: without the flag, requires_grad defaults to False
print("7c) default requires_grad =", torch.tensor([1.0, 3.0]).requires_grad)  # False

print()
print("=" * 60)
print("Question 8: Breaking the graph")
print("=" * 60)

x = torch.tensor([1.0], requires_grad=True)
y = x * 3
z = y.detach()  # z is cut off from the graph: no grad_fn, requires_grad=False
w = z * 2
try:
    w.backward()
except RuntimeError as e:
    print("8) Original code fails:", e)

# Fix: keep z connected to the graph (do not detach)
x = torch.tensor([1.0], requires_grad=True)
y = x * 3
z = y           # still "uses z", but z stays part of the graph
w = z * 2
w.backward()
print("8) Fixed: x.grad =", x.grad)  # dw/dx = 2 * 3 = 6

print()
print("=" * 60)
print("Question 9: Gradient accumulation")
print("=" * 60)

x = torch.tensor([1.0], requires_grad=True)
y1 = x * 2
y1.backward()
print("9) After first backward: x.grad =", x.grad)   # 2
y2 = x * 3
y2.backward()
print("9) After second backward: x.grad =", x.grad)  # 2 + 3 = 5 (accumulated!)

# Fix: reset the gradient before the second backward
x = torch.tensor([1.0], requires_grad=True)
y1 = x * 2
y1.backward()
print("9) Fixed, first backward: x.grad =", x.grad)   # 2
x.grad.zero_()  # in a training loop: optimizer.zero_grad()
y2 = x * 3
y2.backward()
print("9) Fixed, second backward: x.grad =", x.grad)  # 3