# 3D Gaussian Splatting for Real-Time Radiance Field Rendering

# Overview

realtime ≥30fps 1080p novel view synthesis:

1. sparse points from SFM → 3D Gaussians(continuous volumetric rendering)+avoid empty space computation
2. interleaved optimization/density control of 3D Gaussians, optimizing anisotropic covariance (accurate representation of the scene)
3. fast visibility-aware rendering support anisotropic splatting and both accelerates training and allows real time rendering

## 3DGS Method:

1. **differentiable** volumetric representation, **rasterized** very efficiently by **projecting** them to 2D, and applying standard  $\alpha$**-blending**
2. The 3DGS **properties** -3D position, opacity $\alpha$, anisotropic covariance, and spherical harmonic (SH) coefficients – **interleaved with adaptive density control steps**, where we **add** and occasionally **remove** 3D Gaussians during optimization. 
3. fast GPU sorting algorithm
4. anisotropic splatting that respects **visibility order**

## Contribution

1. The introduction of anisotropic 3D Gaussians as a high-quality, unstructured representation of radiance fields.
2. An optimization method of 3D Gaussian properties, interleaved with adaptive density control that creates high-quality representations for captured scenes.
3. A fast, differentiable rendering approach for the GPU, which is visibility-aware, allows anisotropic splatting and fast back-propagation to achieve high-quality novel view synthesis.

![image.png](images/3D%20Gaussian%20Splatting%20for%20Real-Time%20Radiance%20Field%201de71bdab3cf8049a9cec3ca08109157/image.png)

# Method

Maintain conventional $\alpha$ blending on sorted splats: **respect visibility order,**  we back-propagate gradients on **all splats in a pixel** and rasterize **anisotropic** splats. 

![image.png](images/3D%20Gaussian%20Splatting%20for%20Real-Time%20Radiance%20Field%201de71bdab3cf8049a9cec3ca08109157/image%201.png)

![image.png](images/3D%20Gaussian%20Splatting%20for%20Real-Time%20Radiance%20Field%201de71bdab3cf8049a9cec3ca08109157/image%202.png)

## Differentiable 3D Gaussian splatting

high quality novel view generation → **3d gaussian**, differentiable and easily projected to 2D splats allowing alpha blending

![image.png](images/3D%20Gaussian%20Splatting%20for%20Real-Time%20Radiance%20Field%201de71bdab3cf8049a9cec3ca08109157/image%203.png)

![image.png](images/3D%20Gaussian%20Splatting%20for%20Real-Time%20Radiance%20Field%201de71bdab3cf8049a9cec3ca08109157/image%204.png)

this covariance matrix needs to be positive semi-definite (all eigen value ≥0), not always true during training. So we use a scaling matrix R (3D scale)and a Rotation matrix T (a normalized quaternion), to find the covariance.

![image.png](images/3D%20Gaussian%20Splatting%20for%20Real-Time%20Radiance%20Field%201de71bdab3cf8049a9cec3ca08109157/image%205.png)

**derive the gradients** for all parameters **explicitly**

## Optimization

1. able to create geometry, also destory and move geometry
2. SGD+CUDA for fast rasterization(bottle neck)
3. sigmoid for  $\alpha$, exponential activation function for scale of covairance

![image.png](images/3D%20Gaussian%20Splatting%20for%20Real-Time%20Radiance%20Field%201de71bdab3cf8049a9cec3ca08109157/image%206.png)

![image.png](images/3D%20Gaussian%20Splatting%20for%20Real-Time%20Radiance%20Field%201de71bdab3cf8049a9cec3ca08109157/image%207.png)

```python
import torch
import torch.nn.functional as F
import torch.nn as nn

def gaussian_window(window_size, sigma):
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()

def create_window(window_size, channel):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.T
    window = _2D_window.expand(channel, 1, window_size, window_size)
    return window

def ssim(x, y, window_size=11, C1=0.01**2, C2=0.03**2):
    (_, channel, _, _) = x.size()
    window = create_window(window_size, channel).to(x.device)

    mu_x = F.conv2d(x, window, padding=window_size//2, groups=channel)
    mu_y = F.conv2d(y, window, padding=window_size//2, groups=channel)

    sigma_x = F.conv2d(x * x, window, padding=window_size//2, groups=channel) - mu_x ** 2
    sigma_y = F.conv2d(y * y, window, padding=window_size//2, groups=channel) - mu_y ** 2
    sigma_xy = F.conv2d(x * y, window, padding=window_size//2, groups=channel) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))

    return ssim_map.mean()

def dssim(x, y):
    return (1 - ssim(x, y)) / 2

```

![image.png](images/3D%20Gaussian%20Splatting%20for%20Real-Time%20Radiance%20Field%201de71bdab3cf8049a9cec3ca08109157/image%208.png)

## Adaptive Control of Gaussian

![image.png](images/3D%20Gaussian%20Splatting%20for%20Real-Time%20Radiance%20Field%201de71bdab3cf8049a9cec3ca08109157/image%209.png)

1. Populate empty area (under-reconstruction, or over-reconstruction). Large view-space positional gradients. (optimization tries to correct it)
    
    **Densify** Gaussians with an **average magnitude of view-space position** **gradients** above a threshold ??pos, which we set to 0.0002 in our tests.
    
    ![image.png](images/3D%20Gaussian%20Splatting%20for%20Real-Time%20Radiance%20Field%201de71bdab3cf8049a9cec3ca08109157/image%2010.png)
    
    ![image.png](images/3D%20Gaussian%20Splatting%20for%20Real-Time%20Radiance%20Field%201de71bdab3cf8049a9cec3ca08109157/image%2011.png)
    

# Fast differentiable rasterizer for gaussians

1. **pre-sort primitives** for an **entire image** at a time (not sorting each pixel). →fast rasterizer
2. rasterizer: **efficient backpropagation** over **arbitrary number** of blended gaussians. (low memory consumption, **constant overhead** per pixel)
3. rasterizer fully differentiable. projection to 2D can **rasterize anisotropic splats**.

Method:

1. splitting the screen into 16*16 tiles
2. keep Gaussians with a 99% confidence interval intersecting the view frustum.
3. reject Gaussians at extreme positions(too near or too far)
4. instantiate Gaussian(according to the numbers of tiles they overlap, with key (space and tile ID))
    
    ```cpp
    struct GaussianInfo {
    float2 screen_pos;
    float2 scale;         // size in screen space
    int tile_x, tile_y;   // tile coordinates, main tile
    int num_tiles;
    int tile_ids[MAX_TILES]; // list of tiles it overlaps
    };
    ```
    

1. fast GPU Radix sort for all the gaussians
2. each tile has a list of sorted gaussian from nearest to farthest (one thread per tile)
3. for given pixel: accumulate color and $\alpha$ values by traversing the lists from front to back. after arrive **a target saturation** of $\alpha$ (only rule of termination), stop

![image.png](images/3D%20Gaussian%20Splatting%20for%20Real-Time%20Radiance%20Field%201de71bdab3cf8049a9cec3ca08109157/image%2012.png)

![image.png](images/3D%20Gaussian%20Splatting%20for%20Real-Time%20Radiance%20Field%201de71bdab3cf8049a9cec3ca08109157/image%2013.png)

1. During the **backward pass**, instead of storing long per-pixel lists of Gaussians (which would require dynamic memory), the system **re-traverses the sorted tile-wise Gaussian lists** from the forward pass, now in **back-to-front** (far-to-near) order.
    
    To optimize performance:
    
    - **Shared memory** is reused to collaboratively load Gaussians within each tile.
    - Each **pixel only processes Gaussians whose depth is ≤ the depth of the last contributing Gaussian** from the forward pass — avoiding expensive overlap tests for occluded Gaussians.
    
    For gradient computation, the system needs the **intermediate opacity** values from the original forward blending process. Instead of storing all intermediate opacities, it:
    
    - Saves only the **final accumulated opacity α** per point during the forward pass.
    - In the backward pass, it **recovers intermediate α values** by dividing the accumulated opacity at a point by that point’s α, enabling correct gradient computation without full opacity history.

## Implementation

1. coarse→fine, 0.25*res →0.5*res →res (update every 250 iteration)
2. SH learning, zero level→ zero, first level→…→four levels (update every 1000 iterations)

A6000 GPU 30k iteration training, 

![image.png](images/3D%20Gaussian%20Splatting%20for%20Real-Time%20Radiance%20Field%201de71bdab3cf8049a9cec3ca08109157/image%2014.png)

we can achieve **state-of-the-art** results even with **random initialization**: we start training from 100K uniformly random Gaussians inside a volume that encloses the scene bounds. 

first pruned to 6k-10k, finally **200k-500k** per scene.

### Ablation

1. initialization from SFM (floater effect)
2. densification: clone(better and faster converge, thin object) and split(clear background)
3. Unlimited depth complexity of splats with gradients. (unstable, severe approximation)
4. Anisotropic Covariance. (anisotropic leads to higher rendering quality)

### Limitations

1. not well observed area has elongated artifacts
2. popping artifacts (requiring a more principled culling approach)<- simple visibility algorithm (Gaussian switching depth/ blending order) & no regularization method
3. Our memory consumption is significantly higher than NeRF-based solutions

This work introduces the first real-time radiance field renderer that achieves high quality and fast training using **3D Gaussians**. It challenges the need for continuous representations and shows strong performance with **splat-based rasterization**. Most of the code is in PyTorch(80%), but full CUDA optimization could make it even faster. Future work may explore mesh reconstruction.
