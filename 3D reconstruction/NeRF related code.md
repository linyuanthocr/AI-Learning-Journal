# NeRF related code

# quadrature rule

![image.png](images/NeRF%20related%20code%201da71bdab3cf806ab59feb0462d54e2f/image.png)

```python
import torch

def volume_render_radiance_field(sigmas, rgbs, deltas):
    """
    Args:
        sigmas: [N_rays, N_samples] - predicted densities at each sample
        rgbs:   [N_rays, N_samples, 3] - predicted colors at each sample
        deltas: [N_rays, N_samples] - distance between adjacent samples
    Returns:
        final_colors: [N_rays, 3] - rendered RGB for each ray
    """

    # 1. Compute alpha = 1 - exp(-sigma * delta)
    alphas = 1.0 - torch.exp(-sigmas * deltas)

    # 2. Compute transmittance T_i = cumprod(1 - alpha) with shifting
    # Add a small epsilon to prevent log(0)
    eps = 1e-10
    transmittance = torch.cumprod(torch.cat([
        torch.ones_like(alphas[:, :1]),  # T_0 = 1
        1.0 - alphas + eps
    ], dim=-1), dim=-1)[:, :-1]  # Shift right

    # 3. Weights = T_i * alpha_i
    weights = transmittance * alphas  # [N_rays, N_samples]

    # 4. Final RGB = weighted sum of colors
    final_colors = torch.sum(weights.unsqueeze(-1) * rgbs, dim=1)  # [N_rays, 3]

    return final_colors

```

# Hierarchical Sampling

![Screenshot 2025-04-18 at 11.49.48 PM.png](images/NeRF%20related%20code%201da71bdab3cf806ab59feb0462d54e2f/Screenshot_2025-04-18_at_11.49.48_PM.png)

![image.png](images/NeRF%20related%20code%201da71bdab3cf806ab59feb0462d54e2f/image%201.png)

✅ PyTorch Code for Inverse Transform Sampling

```python
def sample_pdf(bins, weights, N_samples, eps=1e-5):
    """
    Hierarchical sampling using inverse transform sampling.
    
    Args:
        bins: [B, N_bins+1] bin edges
        weights: [B, N_bins] (unnormalized PDF)
        N_samples: number of samples to draw
    Returns:
        samples: [B, N_samples]
    """
    B, N_bins = weights.shape

    # Normalize weights to get PDF
    pdf = weights + eps  # prevent NaNs
    pdf = pdf / torch.sum(pdf, dim=-1, keepdim=True)

    # Compute CDF
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # prepend 0

    # Draw uniform samples
    u = torch.rand(B, N_samples, device=weights.device)

    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=N_bins)

    # gather edges
    cdf_below = torch.gather(cdf, 1, below)
    cdf_above = torch.gather(cdf, 1, above)
    bins_below = torch.gather(bins, 1, below)
    bins_above = torch.gather(bins, 1, above)

    # linear interpolation
    denom = cdf_above - cdf_below
    denom[denom < eps] = 1  # avoid division by 0
    t = (u - cdf_below) / denom
    samples = bins_below + t * (bins_above - bins_below)

    return samples
```

# Positional encoding

![image.png](images/NeRF%20related%20code%201da71bdab3cf806ab59feb0462d54e2f/image%202.png)

```python
import torch
import torch.nn as nn
import math

class PositionalEncodingNeRF(nn.Module):
    def __init__(self, num_freqs: int = 10, include_input: bool = True):
        """
        Args:
            num_freqs: number of frequency bands (L in the paper)
            include_input: whether to include the original input in the output
        """
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        
        # Frequency bands: [1, 2, 4, ..., 2^(L-1)]
        self.freq_bands = 2.0 ** torch.arange(num_freqs)

    def forward(self, x):
        """
        Args:
            x: [..., D] input coordinates (e.g., 3D position [x,y,z])
        Returns:
            encoded: [..., D * (include_input + 2 * num_freqs)]
        """
        out = [x] if self.include_input else []

        for freq in self.freq_bands:
            out.append(torch.sin(freq * math.pi * x))
            out.append(torch.cos(freq * math.pi * x))

        return torch.cat(out, dim=-1)

```

# MLP code

![image.png](images/NeRF%20related%20code%201da71bdab3cf806ab59feb0462d54e2f/image%203.png)

```python

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeRFOriginal(nn.Module):
    def __init__(self, D=8, W=256, input_ch=60, input_ch_dir=24):
        super(NeRFOriginal, self).__init__()

        self.D = D  # Depth of MLP
        self.W = W  # Width of hidden layers
        self.input_ch = input_ch
        self.input_ch_dir = input_ch_dir

        # Position input (γ(x))
        self.pts_linears = nn.ModuleList()
        self.pts_linears.append(nn.Linear(input_ch, W))
        for i in range(1, D):
            if i == 4:
                # Skip connection: concatenate γ(x)
                self.pts_linears.append(nn.Linear(W + input_ch, W))
            else:
                self.pts_linears.append(nn.Linear(W, W))

        # These two layers come after the 8th layer (NO ReLU after layer 8)
        self.sigma_linear = nn.Linear(W, 1)
        self.feature_linear = nn.Linear(W, W)

        # View-dependent color head
        self.rgb_linear_1 = nn.Linear(W + input_ch_dir, 128)
        self.rgb_linear_2 = nn.Linear(128, 3)

    def forward(self, x):
        input_x, input_d = x
        h = input_x

        # Apply first 8 layers (ReLU after first 7 only)
        for i, l in enumerate(self.pts_linears):
            if i == 4:
                h = torch.cat([h, input_x], dim=-1)
            h = l(h)
            if i < self.D - 1:  # ReLU after layers 0 to 6
                h = F.relu(h)

        # Now h is the output of layer 8 (linear only)

        # σ and feature from h
        sigma = F.relu(self.sigma_linear(h))  # volume density (non-negative)
        feature = self.feature_linear(h)       # feature vector (no activation)

        # Concatenate feature + γ(view dir)
        h = torch.cat([feature, input_d], dim=-1)
        h = F.relu(self.rgb_linear_1(h))
        rgb = torch.sigmoid(self.rgb_linear_2(h))  # RGB in [0,1]

        return rgb, sigma

```

# NeRF train code

---

### ✅ Components to Implement

1. **Model setup** (coarse & fine networks)
2. **Ray generation** (ray origin & direction)
3. **Volume rendering** (coarse & fine samples)
4. **Loss function** (MSE)
5. **Optimizer and training loop**

---

## 🧱 1. MLP Setup

We assume you already have `NeRFOriginal` defined (the corrected MLP above). We'll now define both **coarse and fine models**:

```python
coarse_model = NeRFOriginal().cuda()
fine_model = NeRFOriginal().cuda()

```

---

## 📦 2. Positional Encoding Function

```python
class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs, include_input=True):
        super().__init__()
        self.include_input = include_input
        self.freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)

    def forward(self, x):
        out = [x] if self.include_input else []
        for freq in self.freq_bands:
            out.append(torch.sin(freq * math.pi * x))
            out.append(torch.cos(freq * math.pi * x))
        return torch.cat(out, dim=-1)

```

---

## 🔁 3. Ray Sampling and Rendering Pipeline

Let’s create a helper to render one batch of rays:

```python
def render_rays(ray_origins, ray_directions, model, N_samples,
                pos_enc_fn, dir_enc_fn, near=2.0, far=6.0):
    """
    ray_origins: [B, 3]
    ray_directions: [B, 3]
    model: NeRF MLP
    """

    B = ray_origins.shape[0]

    # 1. Sample points along ray (uniformly)
    t_vals = torch.linspace(0.0, 1.0, N_samples).to(ray_origins.device)
    z_vals = near * (1. - t_vals) + far * t_vals  # [N_samples]
    z_vals = z_vals.expand([B, N_samples])

    pts = ray_origins.unsqueeze(1) + ray_directions.unsqueeze(1) * z_vals.unsqueeze(2)  # [B, N_samples, 3]

    # 2. Positional encoding
    pts_enc = pos_enc_fn(pts.view(-1, 3)).view(B, N_samples, -1)
    dirs_enc = dir_enc_fn(ray_directions)  # [B, dir_enc_dim]
    dirs_enc = dirs_enc.unsqueeze(1).expand(-1, N_samples, -1)

    # 3. Run model
    inputs = (pts_enc.view(-1, pts_enc.shape[-1]), dirs_enc.view(-1, dirs_enc.shape[-1]))
    rgb, sigma = model(inputs)

    rgb = rgb.view(B, N_samples, 3)
    sigma = sigma.view(B, N_samples)

    # 4. Volume rendering (weights & integration)
    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1])
    deltas = torch.cat([deltas, delta_inf], dim=-1)

    alpha = 1.0 - torch.exp(-sigma * deltas)
    T = torch.cumprod(torch.cat([torch.ones((B, 1)).to(alpha.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * T

    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)  # [B, 3]

    return rgb_map, weights, z_vals

```

---

## 🧠 4. Loss & Optimizer

```python
optimizer = torch.optim.Adam(
    list(coarse_model.parameters()) + list(fine_model.parameters()), lr=5e-4
)

loss_fn = nn.MSELoss()

```

---

## 🏋️‍♂️ 5. Full Training Step

```python
def train_step(batch_rays_o, batch_rays_d, target_rgb,
               coarse_model, fine_model, optimizer,
               pos_enc_fn, dir_enc_fn, N_samples=64, N_importance=128):

    # --- Coarse pass ---
    rgb_coarse, weights, z_vals = render_rays(
        batch_rays_o, batch_rays_d, coarse_model,
        N_samples, pos_enc_fn, dir_enc_fn
    )

    # --- Fine sampling (hierarchical) ---
    with torch.no_grad():
        bins = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
        pdf = weights[:, 1:-1] + 1e-5  # avoid 0s
        pdf = pdf / torch.sum(pdf, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)

        u = torch.rand(cdf.shape[0], N_importance).to(cdf.device)
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(inds - 1, 0, cdf.shape[-1] - 1)
        above = torch.clamp(inds, 0, cdf.shape[-1] - 1)

        cdf_below = torch.gather(cdf, 1, below)
        cdf_above = torch.gather(cdf, 1, above)
        bins_below = torch.gather(bins, 1, below)
        bins_above = torch.gather(bins, 1, above)

        denom = cdf_above - cdf_below
        denom[denom < 1e-5] = 1
        t = (u - cdf_below) / denom
        z_samples = bins_below + t * (bins_above - bins_below)

        z_vals_combined, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)

    # --- Fine pass ---
    rgb_fine, _, _ = render_rays(
        batch_rays_o, batch_rays_d, fine_model,
        z_vals_combined.shape[-1], pos_enc_fn, dir_enc_fn,
        near=2.0, far=6.0
    )

    # --- Loss ---
    loss = loss_fn(rgb_coarse, target_rgb) + loss_fn(rgb_fine, target_rgb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

```

---
