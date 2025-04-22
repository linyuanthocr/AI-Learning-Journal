# Instant Neural Graphics Primitives (Instant-NGP)

---

## **🧾 Abstract**

**Instant Neural Graphics Primitives (Instant-NGP)** presents a landmark in neural rendering by achieving **real-time training and rendering of NeRFs**, thanks to innovations across three dimensions:

1. **Fast and adaptive sampling** via **occupancy grids**,
2. **Tiny, fully-fused neural networks** for efficient computation, and
3. A novel **multi-resolution hash encoding** for compact, high-frequency representation.

Together, these components enable training NeRFs in **seconds instead of hours**, speedups of **1000x**!

---

## **🎯 Key Contributions (Three Pillars of Instant-NGP)**

### **1. 📌 Adaptive Sampling with Occupancy Grids**

Traditional NeRF implementations waste compute by sampling:

- In **empty space**, and
- **Behind opaque surfaces**.

**Instant-NGP** solves this with:

- **Multiscale occupancy grids** that mark non-empty voxels,
- Stored as **bitfields**, and
- Updated dynamically throughout training.

A sample is skipped if its occupancy bit is 0. This boosts sampling speed **10x–100x** by focusing only on relevant regions of the scene.

### **2. ⚡ Fully-Fused Tiny Neural Networks**

Querying the MLP has traditionally been a bottleneck in NeRFs. Instant-NGP uses:

- **A 4-layer MLP**, each with **64 neurons**,
- Implemented as a **fully-fused CUDA kernel**, allowing forward + backward passes in one kernel call.

This yields a **5–10× speedup** over standard TensorFlow/PyTorch implementations. The small size is made possible by powerful input encodings (i.e., the hash grid).

### **3. 🧩 Multi-Resolution Hash Encoding (Main Contribution)**

![image.png](images/Instant%20Neural%20Graphics%20Primitives%20(Instant-NGP)%201dd71bdab3cf80da83ddd03edfb4d469/image.png)

Instead of sinusoidal positional encoding, Instant-NGP uses a **trainable, multiresolution hash grid**:

- Input: 3D coordinate x ∈ [0,1]^3
- For each resolution level l = 1…L:
    1. Compute surrounding voxel corners
    2. Hash each corner using a spatial hash
    3. Lookup a trainable F-dimensional feature vector
    4. Linearly interpolate features based on position
- Concatenate interpolated features from all levels

This structure:

- Is **compact** (thanks to hashing),
- Is **trainable** (gradients flow through interpolated features),
- Naturally **handles hash collisions**, since optimization emphasizes relevant keys,
- Allows **dynamic trade-offs** between memory, quality, and performance by tuning:
    - T: hash table size
    - F: feature dimension
    - L: number of levels

---

## **🎨 View-Dependent Color via Spherical Harmonics**

Unlike traditional NeRFs that concatenate view direction to MLP input, Instant-NGP:

- Uses **Spherical Harmonics (SH)** to condition RGB on view direction.
- Predicts 16 **coefficients**

This method is efficient, differentiable, and removes the need for deeper MLPs to model view-dependent effects.

---

---

## **🔢 PyTorch (Simplified) Hash Encoder**

```python
def hash(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i]*primes[i]

    return torch.tensor((1<<log2_hashmap_size)-1).to(xor_result.device) & xor_result
```

```python
class HashEmbedder(nn.Module):
    def __init__(self, bounding_box, n_levels=16, n_features_per_level=2,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(HashEmbedder, self).__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level

        self.b = torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))

        self.embeddings = nn.ModuleList([nn.Embedding(2**self.log2_hashmap_size, \
                                        self.n_features_per_level) for i in range(n_levels)])
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)
            # self.embeddings[i].weight.data.zero_()
        

    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        '''
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex) # B x 3

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = voxel_embedds[:,0]*(1-weights[:,0][:,None]) + voxel_embedds[:,4]*weights[:,0][:,None]
        c01 = voxel_embedds[:,1]*(1-weights[:,0][:,None]) + voxel_embedds[:,5]*weights[:,0][:,None]
        c10 = voxel_embedds[:,2]*(1-weights[:,0][:,None]) + voxel_embedds[:,6]*weights[:,0][:,None]
        c11 = voxel_embedds[:,3]*(1-weights[:,0][:,None]) + voxel_embedds[:,7]*weights[:,0][:,None]

        # step 2
        c0 = c00*(1-weights[:,1][:,None]) + c10*weights[:,1][:,None]
        c1 = c01*(1-weights[:,1][:,None]) + c11*weights[:,1][:,None]

        # step 3
        c = c0*(1-weights[:,2][:,None]) + c1*weights[:,2][:,None]

        return c

    def forward(self, x):
        # x is 3D point position: B x 3
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b**i)
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask = get_voxel_vertices(\
                                                x, self.bounding_box, \
                                                resolution, self.log2_hashmap_size)
            
            voxel_embedds = self.embeddings[i](hashed_voxel_indices)

            x_embedded = self.trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)

        keep_mask = keep_mask.sum(dim=-1)==keep_mask.shape[-1]
        return torch.cat(x_embedded_all, dim=-1), keep_mask
```

---

## **📈 Performance Summary**

| **Component** | **Speedup** | **Notes** |
| --- | --- | --- |
| Adaptive Sampling | 10–100x | Skips empty space via multiscale occupancy |
| Fully-fused MLP | 5–10x | Tiny CUDA MLP, 4 layers × 64 neurons |
| Hash Grid Encoding | ∞ vs Fourier | Compact, trainable, and collision-tolerant |
| **Total NeRF Training Time** | **~1000x** | From hours to **seconds per scene** |

---

## Important modules

```python
# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d
```

```python
class SHEncoder(nn.Module):
    def __init__(self, input_dim=3, degree=4):
    
        super().__init__()

        self.input_dim = input_dim
        self.degree = degree

        assert self.input_dim == 3
        assert self.degree >= 1 and self.degree <= 5

        self.out_dim = degree ** 2

        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        self.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]
        self.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761
        ]

    def forward(self, input, **kwargs):

        result = torch.empty((*input.shape[:-1], self.out_dim), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)

        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                #result[..., 6] = self.C2[2] * (3.0 * zz - 1) # xx + yy + zz == 1, but this will lead to different backward gradients, interesting...
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return result
```

```python
# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
```

```python
import torch

class OccupancyGrid:
    def __init__(self, grid_size=128, decay=0.95, threshold=0.01):
        self.grid_size = grid_size
        self.decay = decay
        self.threshold = threshold

        # 3D grid storing max density per voxel
        self.occupancy = torch.zeros((grid_size,) * 3).float()

    def update(self, density_fn, device='cuda'):
        """
        Update the occupancy grid based on current density field.
        density_fn: function that maps [N, 3] coords → densities
        """
        coords = torch.linspace(0, 1, self.grid_size, device=device)
        grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing='ij'), dim=-1)
        flat_coords = grid.reshape(-1, 3)  # [G^3, 3]

        with torch.no_grad():
            densities = density_fn(flat_coords)  # [G^3]
            densities = densities.reshape(self.grid_size, self.grid_size, self.grid_size)
            self.occupancy = torch.max(self.occupancy * self.decay, densities)

    def is_occupied(self, pts):
        """
        pts: [N, 3] in [0, 1] range
        Returns: mask [N] indicating whether each point is in occupied space
        """
        indices = (pts * self.grid_size).long().clamp(0, self.grid_size - 1)
        occ = self.occupancy[indices[:, 0], indices[:, 1], indices[:, 2]]
        return occ > self.threshold
```

```python
def sample_rays(ray_origins, ray_dirs, occupancy_grid, n_samples=64, near=0.1, far=4.0, device='cuda'):
    """
    ray_origins: [R, 3]
    ray_dirs: [R, 3] (normalized)
    Returns:
      sample_pts: [R, n_samples, 3] - only valid samples
      valid_mask: [R, n_samples] - bool mask
    """
    R = ray_origins.shape[0]

    # Stratified depths [R, n_samples]
    t_vals = torch.linspace(0.0, 1.0, steps=n_samples, device=device)
    z_vals = near * (1. - t_vals) + far * t_vals
    z_vals = z_vals.expand(R, n_samples)

    # Sample points along ray: [R, n_samples, 3]
    sample_pts = ray_origins[:, None, :] + ray_dirs[:, None, :] * z_vals[..., None]

    # Flatten and check occupancy
    flat_pts = sample_pts.reshape(-1, 3)
    mask = occupancy_grid.is_occupied(flat_pts)  # [R * n_samples]
    mask = mask.reshape(R, n_samples)

    return sample_pts, mask
```

```python
# Dummy density function
def dummy_density_fn(xyz):  # xyz: [N, 3]
    return torch.exp(-((xyz - 0.5)**2).sum(dim=-1) * 40)  # blob in center

# Init grid
grid = OccupancyGrid(grid_size=128, decay=0.95)
grid.update(dummy_density_fn)

# Ray example
R = 128
rays_o = torch.rand(R, 3).cuda()
rays_d = torch.randn(R, 3).cuda(); rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

# Sample
pts, mask = sample_rays(rays_o, rays_d, grid, n_samples=64)
```

## **📚 References**

- Müller et al. (2022). [*Instant Neural Graphics Primitives with a Multiresolution Hash Encoding*](https://nvlabs.github.io/instant-ngp/)
- NVIDIA’s [Instant-NGP GitHub Repo](https://github.com/NVlabs/instant-ngp)
- https://github.com/yashbhalgat/HashNeRF-pytorch
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn): CUDA library for fully-fused networks and hash encodings
