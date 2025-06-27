# 2D RoPE: Mathematical Formulation and Principles

## Table of Contents
1. [Introduction and Motivation](#introduction)
2. [1D RoPE Foundation](#1d-rope-foundation)
3. [Extension to 2D: Core Idea](#extension-to-2d)
4. [Mathematical Formulation](#mathematical-formulation)
5. [Key Properties](#key-properties)
6. [Theoretical Advantages](#theoretical-advantages)
7. [Comparison with Alternatives](#comparison)

---

## 1. Introduction and Motivation {#introduction}

### Why 2D RoPE?

Traditional position encodings work well for sequential data (text, audio), but fail to capture the **2D spatial structure** inherent in:
- **Vision Transformers**: Image patches arranged in a 2D grid
- **Document Understanding**: Text with 2D layout
- **Structured Data**: Any data with inherent 2D relationships

### The Core Problem

Consider an image divided into patches:
```
Grid Layout:
[0,0] [0,1] [0,2] [0,3]
[1,0] [1,1] [1,2] [1,3]  
[2,0] [2,1] [2,2] [2,3]
[3,0] [3,1] [3,2] [3,3]
```

**1D encoding** would assign: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15`

**Problem**: Patches `(0,1)` and `(1,0)` get positions `1` and `4`, losing their spatial proximity!

---

## 2. 1D RoPE Foundation {#1d-rope-foundation}

### Basic RoPE Formulation

For a vector **x** at position **m**, RoPE applies rotation in 2D subspaces:

$$
\begin{bmatrix}
x_{2i}^{new} \\
x_{2i+1}^{new}
\end{bmatrix} = 
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix}
\begin{bmatrix}
x_{2i} \\
x_{2i+1}
\end{bmatrix}
$$

Where:
- $\theta_i = \frac{1}{\text{base}^{2i/d}}$ (frequency for dimension pair $i$)
- $m$ is the 1D position
- $d$ is the feature dimension

### Key Insight: Relative Position Encoding

The attention score between positions $m$ and $n$ only depends on their **relative distance** $|m-n|$:

$$
\langle f_q(\mathbf{x}_m, m), f_k(\mathbf{x}_n, n) \rangle = g(\mathbf{x}_m, \mathbf{x}_n, m-n)
$$

---

## 3. Extension to 2D: Core Idea {#extension-to-2d}

### Fundamental Principle

**Split the feature dimension** to encode each spatial coordinate independently:

- **First half** of features: Encode **Y-coordinate** using 1D RoPE
- **Second half** of features: Encode **X-coordinate** using 1D RoPE

### Mathematical Framework

For a 2D position $(y, x)$ and feature vector $\mathbf{h} \in \mathbb{R}^d$:

1. **Split features**: $\mathbf{h} = [\mathbf{h}_y; \mathbf{h}_x]$ where $\mathbf{h}_y, \mathbf{h}_x \in \mathbb{R}^{d/2}$

2. **Apply 1D RoPE separately**:
   - $\mathbf{h}_y^{new} = \text{RoPE}_{1D}(\mathbf{h}_y, y)$
   - $\mathbf{h}_x^{new} = \text{RoPE}_{1D}(\mathbf{h}_x, x)$

3. **Concatenate**: $\mathbf{h}^{new} = [\mathbf{h}_y^{new}; \mathbf{h}_x^{new}]$

---

## 4. Mathematical Formulation {#mathematical-formulation}

### Complete 2D RoPE Formula

For position $(y, x)$ and feature vector $\mathbf{h} = [h_0, h_1, \ldots, h_{d-1}]$:

#### Y-coordinate encoding (first $d/2$ dimensions):
$$
\begin{bmatrix}
h_{2i}^{y,new} \\
h_{2i+1}^{y,new}
\end{bmatrix} = 
\begin{bmatrix}
\cos(y\theta_i) & -\sin(y\theta_i) \\
\sin(y\theta_i) & \cos(y\theta_i)
\end{bmatrix}
\begin{bmatrix}
h_{2i} \\
h_{2i+1}
\end{bmatrix}
$$

#### X-coordinate encoding (second $d/2$ dimensions):
$$
\begin{bmatrix}
h_{d/2+2j}^{x,new} \\
h_{d/2+2j+1}^{x,new}
\end{bmatrix} = 
\begin{bmatrix}
\cos(x\theta_j) & -\sin(x\theta_j) \\
\sin(x\theta_j) & \cos(x\theta_j)
\end{bmatrix}
\begin{bmatrix}
h_{d/2+2j} \\
h_{d/2+2j+1}
\end{bmatrix}
$$

Where:
- $\theta_i = \frac{1}{\text{base}^{2i/(d/2)}}$ for $i = 0, 1, \ldots, d/4-1$
- $\theta_j = \frac{1}{\text{base}^{2j/(d/2)}}$ for $j = 0, 1, \ldots, d/4-1$

### Frequency Base Selection

Unlike text (base = 10,000), images use **smaller base values**:
- **Text RoPE**: base = 10,000 (long sequences)
- **2D RoPE**: base = 100-200 (small spatial grids)

**Reasoning**: 
- Image patches typically in 14×14 to 32×32 grids
- Smaller base → higher frequencies → better spatial resolution

---

## 5. Key Properties {#key-properties}

### 5.1 Relative Position Invariance

**Property**: Attention between patches $(y_1, x_1)$ and $(y_2, x_2)$ depends only on **relative displacement** $(Δy, Δx) = (y_2-y_1, x_2-x_1)$.

**Mathematical Proof**:
Let $\mathbf{q}_1, \mathbf{k}_2$ be query/key vectors after 2D RoPE encoding.

$$
\mathbf{q}_1^T \mathbf{k}_2 = f(\mathbf{q}_1^{raw}, \mathbf{k}_2^{raw}, y_2-y_1, x_2-x_1)
$$

This means patches with the **same relative displacement** have similar attention patterns!

### 5.2 Translation Equivariance

**Property**: Shifting the entire grid by $(Δy, Δx)$ produces equivalent attention patterns.

Formally: $\text{Attention}(\text{Grid}) = \text{Attention}(\text{Grid} + (Δy, Δx))$

### 5.3 Zero Parameter Overhead

Unlike learned positional embeddings, 2D RoPE requires **no additional parameters**:
- No embedding tables
- No trainable position vectors
- Pure geometric transformation

### 5.4 Length Extrapolation

Can handle **larger grids** than seen during training:
- Train on 14×14 patches
- Inference on 32×32 patches ✅

---

## 6. Theoretical Advantages {#theoretical-advantages}

### 6.1 Spatial Structure Preservation

**Traditional approaches** (absolute position embedding):
```
Position: [0] [1] [2] [3] [4] [5] [6] [7] [8] ...
Grid:     [0,0][0,1][0,2][1,0][1,1][1,2][2,0][2,1][2,2] ...
```
❌ **Problem**: Adjacent grid positions get distant linear positions

**2D RoPE**:
```
Position: (0,0) (0,1) (0,2) (1,0) (1,1) (1,2) (2,0) (2,1) (2,2)
```
✅ **Solution**: Directly encodes 2D coordinates

### 6.2 Inductive Bias for Vision

**Horizontal/Vertical Relationships**: 
- Patches in same row share Y-coordinate encoding
- Patches in same column share X-coordinate encoding
- Natural bias for grid-structured data

**Diagonal Relationships**:
- Captured through combination of X and Y encodings
- Preserves 2D geometric relationships

### 6.3 Computational Efficiency

**Memory Complexity**:
- Absolute PE: $O(H \times W \times d)$ parameters
- 2D RoPE: $O(1)$ parameters (zero!)

**Computation Complexity**:
- Same as standard attention: $O(n^2 d)$
- No additional matrix multiplications

---

## 7. Comparison with Alternatives {#comparison}

### 7.1 vs. Absolute 2D Position Embedding

| Aspect | Absolute 2D PE | 2D RoPE |
|--------|----------------|----------|
| **Parameters** | $O(H \times W \times d)$ | $O(1)$ |
| **Extrapolation** | ❌ Fixed grid size | ✅ Any grid size |
| **Relative Position** | ❌ Not explicitly modeled | ✅ Built-in |
| **Translation Invariance** | ❌ Position-dependent | ✅ Natural property |

### 7.2 vs. Relative Position Embedding

| Aspect | Relative 2D PE | 2D RoPE |
|--------|----------------|----------|
| **Parameters** | $O(H \times W \times \text{heads})$ | $O(1)$ |
| **Implementation** | Complex attention modification | Simple feature transformation |
| **Memory** | $O(n^2)$ attention bias | $O(1)$ |
| **Efficiency** | Slower (bias computation) | Faster (direct encoding) |

### 7.3 vs. Learned 2D Embeddings

| Aspect | Learned 2D | 2D RoPE |
|--------|------------|----------|
| **Flexibility** | ✅ Can learn any pattern | ❌ Fixed geometric pattern |
| **Generalization** | ❌ Dataset-specific | ✅ Universal geometric |
| **Interpretability** | ❌ Black box | ✅ Clear geometric meaning |
| **Robustness** | ❌ Overfitting risk | ✅ Principled approach |

```python
class RoPE2D(torch.nn.Module):
        
        def __init__(self, freq=100.0, F0=1.0):
            super().__init__()
            self.base = freq 
            self.F0 = F0
            self.cache = {}

        def get_cos_sin(self, D, seq_len, device, dtype):
            if (D,seq_len,device,dtype) not in self.cache:
                inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
                t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
                freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
                freqs = torch.cat((freqs, freqs), dim=-1)
                cos = freqs.cos() # (Seq, Dim)
                sin = freqs.sin()
                self.cache[D,seq_len,device,dtype] = (cos,sin)
            return self.cache[D,seq_len,device,dtype]
            
        @staticmethod
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
            
        def apply_rope1d(self, tokens, pos1d, cos, sin):
            assert pos1d.ndim==2
            cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
            sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
            return (tokens * cos) + (self.rotate_half(tokens) * sin)
            
        def forward(self, tokens, positions):
            """
            input:
                * tokens: batch_size x nheads x ntokens x dim
                * positions: batch_size x ntokens x 2 (y and x position of each token)
            output:
                * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
            """
            assert tokens.size(3)%2==0, "number of dimensions should be a multiple of two"
            D = tokens.size(3) // 2
            assert positions.ndim==3 and positions.shape[-1] == 2 # Batch, Seq, 2
            cos, sin = self.get_cos_sin(D, int(positions.max())+1, tokens.device, tokens.dtype)
            # split features into two along the feature dimension, and apply rope1d on each half
            y, x = tokens.chunk(2, dim=-1)
            y = self.apply_rope1d(y, positions[:,:,0], cos, sin)
            x = self.apply_rope1d(x, positions[:,:,1], cos, sin)
            tokens = torch.cat((y, x), dim=-1)
            return tokens
```            
---

## Summary

### Core Insights

1. **Geometric Principle**: 2D RoPE directly encodes 2D spatial relationships through rotation
2. **Efficiency**: Zero parameters, same computational cost as standard attention
3. **Generalization**: Works for any grid size, maintains relative position properties
4. **Simplicity**: Clean mathematical formulation with clear geometric interpretation


### Mathematical Beauty

2D RoPE elegantly extends the 1D rotation concept to 2D space while preserving all the desirable properties of the original RoPE formulation. It's a perfect example of how mathematical elegance can lead to practical improvements in machine learning systems.
