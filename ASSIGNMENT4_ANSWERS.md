# Assignment 4: Diffusion and Energy-Based Models

## Part 1: Model deployment

Both models are trained on **CIFAR-10** (32×32 RGB) and served by the FastAPI app.

| File | What it does |
|---|---|
| `app/diffusion_model.py` | Offset cosine schedule, sinusoidal embedding, U-Net, `DiffusionModel` (EMA, reverse diffusion) |
| `app/ebm_model.py` | `EnergyModel` CNN and `generate_samples` (Langevin dynamics on the input image) |
| `train_diffusion.py` | Trains the diffusion model and saves `app/diffusion_weights.pth` |
| `train_ebm.py` | Trains the EBM with contrastive divergence and a replay buffer, then saves `app/ebm_weights.pth` |
| `app/main.py` | New endpoints `GET /generate-diffusion` and `GET /generate-ebm` (return a PNG grid) |

## Part 2: Building blocks of a diffusion model

### Question 1: Sinusoidal embedding

The timestep `t` (a single number) is turned into a vector of dimension `d` by evaluating sines and cosines at **different frequencies**. Low frequencies change slowly with `t`, and high frequencies change quickly, so every `t` gets a unique, smooth "fingerprint" that the network can use.

For `i = 0, 1, …, d/2 − 1`, with max period `P = 10000`:

```
ω_i = 1 / P^(2i/d)

PE(t, 2i)   = sin(ω_i · t) = sin( t / 10000^(2i/d) )
PE(t, 2i+1) = cos(ω_i · t) = cos( t / 10000^(2i/d) )
```

(Many diffusion implementations, including our class code, concatenate all the sines first and then all the cosines: `[sin(ω_0 t), …, sin(ω_{d/2−1} t), cos(ω_0 t), …, cos(ω_{d/2−1} t)]`. It's the same information in a different order.)

### Question 2: Embedding for d = 8, t = 1, P = 10000

`d/2 = 4` frequencies, with `i = 0, 1, 2, 3`:

| i | 2i/d | ω_i = 10000^(−2i/d) | ω_i · t |
|---|---|---|---|
| 0 | 0 | 1 | 1 |
| 1 | 0.25 | 10000^−0.25 = 0.1 | 0.1 |
| 2 | 0.5 | 10000^−0.5 = 0.01 | 0.01 |
| 3 | 0.75 | 10000^−0.75 = 0.001 | 0.001 |

| i | sin(ω_i) | cos(ω_i) |
|---|---|---|
| 0 | sin(1) = 0.8415 | cos(1) = 0.5403 |
| 1 | sin(0.1) = 0.0998 | cos(0.1) = 0.9950 |
| 2 | sin(0.01) = 0.0100 | cos(0.01) = 0.99995 ≈ 1.0000 |
| 3 | sin(0.001) = 0.0010 | cos(0.001) = 0.9999995 ≈ 1.0000 |

**Interleaved (transformer) order** `[sin, cos, sin, cos, …]`:

```
[0.8415, 0.5403, 0.0998, 0.9950, 0.0100, 1.0000, 0.0010, 1.0000]
```

**Concatenated (diffusion code) order** `[sin…, cos…]`:

```
[0.8415, 0.0998, 0.0100, 0.0010, 0.5403, 0.9950, 1.0000, 1.0000]
```

Check in code:

```python
import math
d, P, t = 8, 10000, 1
freqs = [P ** (-2 * i / d) for i in range(d // 2)]           # [1, 0.1, 0.01, 0.001]
emb = [f(w * t) for w in freqs for f in (math.sin, math.cos)]
print([round(v, 4) for v in emb])
# [0.8415, 0.5403, 0.0998, 0.995, 0.01, 1.0, 0.001, 1.0]
```

### Question 3: Positional encoding vs. sinusoidal time embedding

**Same math:** both use the same sine/cosine formula at geometrically spaced frequencies to turn an integer into a smooth, unique vector that a network can use.

**Key difference in how they are used:**
- **Transformers:** the input is the **position of a token in a sequence**. There is one encoding *per token*. It is **added** to each token's embedding so attention, which is otherwise order-blind, knows word order.
- **Diffusion models:** the input is the **timestep / noise level** `t`. There is **one embedding for the whole image**, shared by every pixel. It is **broadcast across the spatial dimensions** and concatenated (or, in other U-Nets, passed through an MLP and added) inside the residual blocks. It tells the U-Net *how noisy* the image is, so the same network can denoise at every noise level.

In short: transformers encode **"where"** (position in a sequence), and diffusion models encode **"when"** (how far along the noising process we are).

### Question 4: Bottleneck resolution

Each stride-2 downsampling halves the height and width:

```
64×64 → (÷2) 32×32 → (÷2) 16×16 → (÷2) 8×8
```

The bottleneck is **8×8** (in general, `64 / 2³ = 8`).

### Question 5: U-Net output and loss

The U-Net receives the noisy image `x_t` and the timestep `t` (in our code, the noise variance), and outputs **its prediction of the noise** `ε̂ = ε_θ(x_t, t)`, a tensor the same shape as the image.

During training:
1. Take a clean image `x_0`, sample noise `ε ~ N(0, I)` and a random `t`.
2. Make the noisy image: `x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε` (in our code: `signal_rate · x_0 + noise_rate · ε`).
3. The U-Net predicts `ε̂ = ε_θ(x_t, t)`.
4. The loss compares the **true noise** with the **predicted noise**:

```
L = || ε − ε_θ(x_t, t) ||²      (MSE, as in DDPM)
```

Our class code uses the L1 version, `L = mean |ε − ε̂|` (`nn.L1Loss()`). The predicted noise can also be used to recover the clean image: `x̂_0 = (x_t − noise_rate · ε̂) / signal_rate`. That's what reverse diffusion uses at generation time.

## Part 3: Building blocks of an energy model

All the code below is in `assignment4_part3.py`. Run it with `uv run python assignment4_part3.py`.

### Question 6: Basic gradient calculations

**a)** `y = x² + 3x`, so `dy/dx = 2x + 3`. At `x = 2`: `2(2) + 3 = 7`.
Output: `x.grad = tensor([7.])`

**b)** With `requires_grad=False`, PyTorch doesn't build a computational graph, so `y` has no `grad_fn` and `y.backward()` raises:
`RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

**c)** No. According to the `torch.tensor` documentation, `requires_grad` defaults to **`False`**, so gradients aren't tracked unless you ask for them.

### Question 7: Introduce weights

**a)** Output:

```
x.grad = tensor([7.])     # dy/dx = 2·w0·x + w1 = 2·1·2 + 3 = 7
w.grad = None
```

`w.grad` is `None` because `w` was created without `requires_grad=True`. PyTorch treats `w` as a constant, so it doesn't record how `y` depends on `w` and computes no gradient for it.

**b)** Create `w` with `requires_grad=True`:

```python
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([1.0, 3.0], requires_grad=True)
y = w[0] * x**2 + w[1] * x
y.backward()
print("x.grad =", x.grad)   # tensor([7.])
print("w.grad =", w.grad)   # tensor([4., 2.])
```

By hand: `∂y/∂w0 = x² = 4` and `∂y/∂w1 = x = 2`, so `w.grad = [4, 2]`.

**c)** No. As in 6c, `requires_grad` defaults to `False`. That's exactly why `w.grad` was `None` in part (a).

### Question 8: Breaking the graph

`y.detach()` returns a tensor with the same values that is **cut off from the computational graph** (`requires_grad=False`, no `grad_fn`). So `w = z * 2` has no history back to `x`, and `w.backward()` raises
`RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`.

**Fix:** keep `z` connected to the graph by not detaching:

```python
x = torch.tensor([1.0], requires_grad=True)
y = x * 3
z = y          # no .detach()
w = z * 2
w.backward()
print(x.grad)  # tensor([6.])  ->  dw/dx = 2 · 3 = 6
```

(Note: `y.detach().requires_grad_()` would remove the error but **not** let gradients reach `x`. It creates a new leaf, so `x.grad` would stay `None`.)

### Question 9: Gradient accumulation

Output:

```
After first backward:  x.grad = tensor([2.])
After second backward: x.grad = tensor([5.])
```

PyTorch **accumulates** (adds) gradients into `.grad` rather than overwriting them. The second call adds `d(3x)/dx = 3` to the stored `2`, giving `5` instead of `3`.

**To avoid it,** reset the gradient before each new backward pass:

```python
x.grad.zero_()           # or x.grad = None
# in a training loop:
optimizer.zero_grad()
```

This is why every training loop calls `optimizer.zero_grad()` before `loss.backward()`. In Langevin sampling for EBMs, we use `torch.autograd.grad(...)`, which *returns* the gradient instead of accumulating it into `.grad`.