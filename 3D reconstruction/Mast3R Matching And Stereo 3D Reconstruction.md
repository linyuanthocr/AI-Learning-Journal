# Mast3R: Matching And Stereo 3D Reconstruction

https://arxiv.org/abs/2406.09756

[https://github.com/naver/mast3r](https://github.com/naver/mast3r)

Binocular Case: matching & 3D reconstruction.

![image.png](images/Mast3R%20Matching%20And%20Stereo%203D%20Reconstruction%201e171bdab3cf80af9ad8fe842e9b4c83/image.png)

## Contributions

1. 3D matching aware (with local features)
2. coarse to fine, high resolution matching
3. state of art in localization (absolution and relative error)

# Method

![image.png](images/Mast3R%20Matching%20And%20Stereo%203D%20Reconstruction%201e171bdab3cf80af9ad8fe842e9b4c83/image%201.png)

## Dust3R Framework:

 [**DUSt3R: Geometric 3D Vision Made Easy**](https://www.notion.so/DUSt3R-Geometric-3D-Vision-Made-Easy-1ba71bdab3cf80a08e7afbedcb4a1605?pvs=21) 

The Dust3R loss is as follows, **normalizing factors z and z^** are introduced to make the reconstruction **invariant to scale**.

![image.png](images/Mast3R%20Matching%20And%20Stereo%203D%20Reconstruction%201e171bdab3cf80af9ad8fe842e9b4c83/image%202.png)

now it changes into:

![image.png](images/Mast3R%20Matching%20And%20Stereo%203D%20Reconstruction%201e171bdab3cf80af9ad8fe842e9b4c83/image%203.png)

## Matching prediction head and loss

DUSt3R matching is suboptimal： (i) regression is inherently affected by noise, and (ii) because DUSt3R was never explicitly trained for matching. (H1 is the encoder output, and H1’ is the decoder output)

### Matching head

![image.png](images/Mast3R%20Matching%20And%20Stereo%203D%20Reconstruction%201e171bdab3cf80af9ad8fe842e9b4c83/image%204.png)

2 layers of (MLP+GELU) + normalized each feature to unit norm.

![image.png](images/Mast3R%20Matching%20And%20Stereo%203D%20Reconstruction%201e171bdab3cf80af9ad8fe842e9b4c83/image%205.png)

![image.png](images/Mast3R%20Matching%20And%20Stereo%203D%20Reconstruction%201e171bdab3cf80af9ad8fe842e9b4c83/image%206.png)

```python
x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)  # small epsilon to avoid division by zero
```

### Match projective

Encourage **each local descriptor** from one image to match with **at most a single** descriptor from the other image that represents the **same 3D point** in the scene.

![image.png](images/Mast3R%20Matching%20And%20Stereo%203D%20Reconstruction%201e171bdab3cf80af9ad8fe842e9b4c83/image%207.png)

![image.png](images/Mast3R%20Matching%20And%20Stereo%203D%20Reconstruction%201e171bdab3cf80af9ad8fe842e9b4c83/image%208.png)

**infoNCE**

![image.png](images/Mast3R%20Matching%20And%20Stereo%203D%20Reconstruction%201e171bdab3cf80af9ad8fe842e9b4c83/image%209.png)

```python
def get_similarities(desc1, desc2, euc=False):
    if euc:  # euclidean distance in same range than similarities
        dists = (desc1[:, :, None] - desc2[:, None]).norm(dim=-1)
        sim = 1 / (1 + dists)
    else:
        # Compute similarities
        sim = desc1 @ desc2.transpose(-2, -1)
    return sim

class InfoNCE(MatchingCriterion):
    def __init__(self, temperature=0.07, eps=1e-8, mode='all', **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.eps = eps
        assert mode in ['all', 'proper', 'dual']
        self.mode = mode

    def loss(self, desc1, desc2, valid_matches=None, euc=False):
        # valid positives are along diagonals
        B, N, D = desc1.shape
        B2, N2, D2 = desc2.shape
        assert B == B2 and D == D2
        if valid_matches is None:
            valid_matches = torch.ones([B, N], dtype=bool)
        # torch.all(valid_matches.sum(dim=-1) > 0) some pairs have no matches????
        assert valid_matches.shape == torch.Size([B, N]) and valid_matches.sum() > 0

        # Tempered similarities
        sim = get_similarities(desc1, desc2, euc) / self.temperature
        sim[sim.isnan()] = -torch.inf  # ignore nans
        # Softmax of positives with temperature
        sim = sim.exp_()  # save peak memory
        positives = sim.diagonal(dim1=-2, dim2=-1)

        # Loss
        if self.mode == 'all':            # Previous InfoNCE
            loss = -torch.log((positives / sim.sum(dim=-1).sum(dim=-1, keepdim=True)).clip(self.eps))
        elif self.mode == 'proper':  # Proper InfoNCE
            loss = -(torch.log((positives / sim.sum(dim=-2)).clip(self.eps)) +
                     torch.log((positives / sim.sum(dim=-1)).clip(self.eps)))
        elif self.mode == 'dual':  # Dual Softmax
            loss = -(torch.log((positives**2 / sim.sum(dim=-1) / sim.sum(dim=-2)).clip(self.eps)))
        else:
            raise ValueError("This should not happen...")
        return loss[valid_matches]
```

## Fast reciprocal matching

- **Subsample** k sparse points from image 1 (regular grid).
- For each point:
    - **NN match** to image 2 (using feature descriptors).
    - **NN match back** to image 1.
- **Reciprocal matches**: keep points that cycle back correctly. (From the 2 to 1, this time, alternately)
- **Filter out** matched points (converged).
- **Iterate** on the remaining unmatched points (few rounds).
- **Concatenate** all reciprocal matches from all iterations.

✅ No 3D involved yet, **only feature matching**.

✅ Iteration = shrinking the set of unmatched points.

![image.png](images/Mast3R%20Matching%20And%20Stereo%203D%20Reconstruction%201e171bdab3cf80af9ad8fe842e9b4c83/image%2010.png)

![image.png](images/Mast3R%20Matching%20And%20Stereo%203D%20Reconstruction%201e171bdab3cf80af9ad8fe842e9b4c83/image%2011.png)

![image.png](images/Mast3R%20Matching%20And%20Stereo%203D%20Reconstruction%201e171bdab3cf80af9ad8fe842e9b4c83/image%2012.png)

![image.png](images/Mast3R%20Matching%20And%20Stereo%203D%20Reconstruction%201e171bdab3cf80af9ad8fe842e9b4c83/image%2013.png)

## Coarse to Fine matching (Feature matching)

![image.png](images/Mast3R%20Matching%20And%20Stereo%203D%20Reconstruction%201e171bdab3cf80af9ad8fe842e9b4c83/image%2014.png)

![image.png](images/Mast3R%20Matching%20And%20Stereo%203D%20Reconstruction%201e171bdab3cf80af9ad8fe842e9b4c83/image%2015.png)

# Experimental results

## Training

**Data**

14 datasets:Habitat [74], ARKitScenes [20], Blended MVS [112], MegaDepth [48], Static Scenes 3D [57],ScanNet++ [113], CO3D-v2 [67], Waymo [83], Map- free [5], WildRgb [2], VirtualKitti [12], Unreal4K [91], TartanAir [103] and an internal dataset. Covers indoor, outdoor, synthetic, real-world, object-centric, etc. Among them, 10 datasets have metric ground-truth. 

Specifically, we utilize off-the-shelf image retrieval and point matching algorithms to **match and verify image pairs**.

**Training**

checkpoint of DUSt3R (Vit-L encoder & Vit-B decoder). 65k equally distributed image pair per iteration.

**Correspondences**

![image.png](images/Mast3R%20Matching%20And%20Stereo%203D%20Reconstruction%201e171bdab3cf80af9ad8fe842e9b4c83/image%2016.png)

![image.png](images/Mast3R%20Matching%20And%20Stereo%203D%20Reconstruction%201e171bdab3cf80af9ad8fe842e9b4c83/image%2017.png)

![image.png](images/Mast3R%20Matching%20And%20Stereo%203D%20Reconstruction%201e171bdab3cf80af9ad8fe842e9b4c83/image%2018.png)