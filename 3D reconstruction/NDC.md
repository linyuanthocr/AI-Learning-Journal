# Normalized Device Coordinates_NDC

# 📐 Deriving NeRF's NDC Projection (Normalized Device Coordinates)

This is a typical **perspective projection** formula, similar to the NDC transformation used in OpenGL. It is used in the original NeRF implementation to map 3D points from camera space into a normalized volume for more stable training.

---

## 🎯 Goal

Map a 3D point (x, y, z) from camera space into **NDC** space (x_ndc, y_ndc, z_ndc).

---

## 🧾 Assumptions

- Image width: W, height:H
- Focal length: f
- Point lies in camera space: (x, y, z)
- z is negative (since the camera looks down the negative z-axis)
- Near plane at depth = `near`

---

![image.png](images/NDC%201dd71bdab3cf800d85b1dcbbd95bce6d/image.png)

![image.png](images/NDC%201dd71bdab3cf800d85b1dcbbd95bce6d/199b66f5-1ff6-4b64-b849-4f7d5841e772.png)

---

### 2. Project z to NDC Depth

This is the NDC-space version of the 3D point after transforming from camera space.

---

## 🔁 Code Equivalent (PyTorch)

```python
o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]  # x_ndc
o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]  # y_ndc
o2 = 1. + 2. * near / rays_o[..., 2]                              # z_ndc
```

Let's now **derive the NDC-transformed ray direction (`rays_d`)** — this part is a little trickier than the origin, but very insightful.

---

### 🎯 Goal

We want to find the **direction** of the ray **after projection into NDC space**, given:

- The ray **origin** is already moved to the near plane and projected to NDC: `rays_o_ndc`
- The ray **direction** is originally a vector `rays_d` in camera/world space

We aim to derive a new direction vector `rays_d_ndc`.

---

### 📌 Insight: Why not just apply the same projection to `rays_d`?

Because projection (like in OpenGL) is **nonlinear in z** — it divides by `z`. That means you **can’t directly apply the same projection to a direction vector**, since a direction doesn’t have a meaningful position (no fixed origin).

---

### ✅ Trick used in NeRF

Instead of directly projecting the direction, we compute it **as the difference between two projected points**:

Let:

- `rays_o` = origin on the near plane (already projected to NDC)
- `rays_o + rays_d` = a second point along the ray
- Project both to NDC space, then subtract their NDC positions to get a direction vector

---

### 🧮 Step-by-step

Let’s take a second point along the ray:

```python
P = rays_o + rays_d

```

Then compute:

```python
projected_P = perspective_project(P)
projected_rays_o = perspective_project(rays_o)
rays_d_ndc = projected_P - projected_rays_o

```

That’s the idea behind this NeRF code:

```python
d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
d2 = -2. * near / rays_o[...,2]

```

---

### ✅ Explanation of each term

### d0 (x direction):

```python
rays_d_ndc[...,0] = -1 / (W / (2 * focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])

```

![image.png](images/NDC%201dd71bdab3cf800d85b1dcbbd95bce6d/image%201.png)

---

### 📌 Summary

The NDC direction vector is computed as:

```python
rays_d_ndc = projected_point - projected_origin

```

Where both are computed using the same perspective projection logic.

---

### ✅ Why this works

- Projection is **nonlinear**, but subtracting **two projected points** approximates the local ray direction in NDC space
- Ensures the rays stay accurate for **volume sampling in NDC coordinates**

---

# NDC_rays code

```python
def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d
```

 **NDC projection is a nonlinear transformation that compresses distant geometry and expands nearby geometry**, effectively making **nearby surfaces occupy more volume in NDC space** — and thus receive **more samples** during rendering.

![image.png](images/NDC%201dd71bdab3cf800d85b1dcbbd95bce6d/image%202.png)

![image.png](images/NDC%201dd71bdab3cf800d85b1dcbbd95bce6d/image%203.png)
