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

### Matching head

![image.png](images/Mast3R%20Matching%20And%20Stereo%203D%20Reconstruction%201e171bdab3cf80af9ad8fe842e9b4c83/image%204.png)

DUSt3R matching is suboptimal： (i) regression is inherently affected by noise, and (ii) because DUSt3R was never explicitly trained for matching. (H1 is the encoder output, and H1’ is the decoder output)

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

if the input too large, it use blockwise calculation, small chunks to do this:

```python
    def blockwise_criterion(self, descs1, descs2, confs1, confs2, valid_matches, euc, rng=np.random, shuffle=True):
        loss = None
        details = {}
        B, N, D = descs1.shape

        if N <= self.blocksize:  # Blocks are larger than provided descs, compute regular loss
            loss = self.criterion(descs1, descs2, valid_matches, euc=euc)
        else:  # Compute criterion on the blockdiagonal only, after shuffling
            # Shuffle if necessary
            matches_perm = slice(None)
            if shuffle:
                matches_perm = np.stack([rng.choice(range(N), size=N, replace=False) for _ in range(B)])
                batchid = torch.tile(torch.arange(B), (N, 1)).T
                matches_perm = batchid, matches_perm

            descs1 = descs1[matches_perm]
            descs2 = descs2[matches_perm]
            valid_matches = valid_matches[matches_perm]

            assert N % self.blocksize == 0, "Error, can't chunk block-diagonal, please check blocksize"
            n_chunks = N // self.blocksize
            descs1 = descs1.reshape([B * n_chunks, self.blocksize, D])  # [B*(N//blocksize), blocksize, D]
            descs2 = descs2.reshape([B * n_chunks, self.blocksize, D])  # [B*(N//blocksize), blocksize, D]
            valid_matches = valid_matches.view([B * n_chunks, self.blocksize])
            loss = self.criterion(descs1, descs2, valid_matches, euc=euc)
            if self.withconf:
                confs1, confs2 = map(lambda x: x[matches_perm], (confs1, confs2))  # apply perm to confidences if needed

        if self.withconf:
            # split confidences between positives/negatives for loss computation
            details['conf_pos'] = map(lambda x: x[valid_matches.view(B, -1)], (confs1, confs2))
            details['conf_neg'] = map(lambda x: x[~valid_matches.view(B, -1)], (confs1, confs2))
            details['Conf1_std'] = confs1.std()
            details['Conf2_std'] = confs2.std()

        return loss, details
```

## Fast reciprocal matching

- **Subsample** k sparse points from image 1 (regular grid).
- For each point:
    - **NN match** to image 2 (using feature descriptors).
    - **NN match back** to image 1.
    
    **Data**
    
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

1. grid sample
2. make sure it’s normalized (in the boundary, aspect ratio, etc)
3. score cells：
    
    **Purpose:**
    
    Estimates how well a crop (cell) in one image aligns with a corresponding crop in another image based on matched keypoints.
    
    **Key Steps:**
    
    - Assign keypoints to each crop.
    - Filter out crops with too few correspondences.
    - Compute the geometric center and scale (robust std) of assigned points.
    - Predict the corresponding crop in the second image.
    - Calculate correspondence weights using spatial proximity. (gaussian distance)
    
    **Output:**
    
    Returns crop pairs (cell1, cell2) and a weight matrix for matched keypoin
    
4. greedy selection
    
    **Purpose:**
    
    Greedily selects the smallest set of crop pairs to cover a target percentage (e.g., 90%) of the total match weight.
    
    **Key Steps:**
    
    - Compute the maximum achievable total match weight.
    - Iteratively pick the crop pair that contributes the most remaining weight.
    - Update the accumulated coverage.
    - Suppress overlapping contribution to avoid double counting.
    
    **Output:**
    
    Returns indices of the selected crop pairs.
    
5. fine extraction+concatenate

```python
#cell1 shape: (4, N_cell, 1)
#p1 shape: (N_points, 2)
#assigned shape: (N_cell, N_points)
def pos2d_in_rect(p1, cell1):
    x, y = p1.T # x.shape==(N_points,)
    l, t, r, b = cell1 #l.shape = (N_cell,1)
    assigned = (l <= x) & (x < r) & (t <= y) & (y < b)
    return assigned

def _weight_pixels(cell, pix, assigned, gauss_var=2):
    center = cell.reshape(-1, 2, 2).mean(axis=1)
    width, height = _cell_size(cell)

    # square distance between each cell center and each point
    dist = (center[:, None] - pix[None]) / np.c_[width, height][:, None]
    dist2 = np.square(dist).sum(axis=-1)

    assert assigned.shape == dist2.shape
    res = np.where(assigned, np.exp(-gauss_var * dist2), 0)
    return res
    
def _score_cell(cell1, H2, W2, p1, p2, min_corres=10, forced_resolution=None):
    assert p1.shape == p2.shape

    # compute keypoint assignment
    assigned = pos2d_in_rect(p1, cell1[None].T)
    assert assigned.shape == (len(cell1), len(p1))

    # remove cells without correspondences
    valid_cells = assigned.sum(axis=1) >= min_corres
    cell1 = cell1[valid_cells]
    assigned = assigned[valid_cells]
    if not valid_cells.any():
        return cell1, cell1, assigned

    # fill-in the assigned points in both image
    assigned_p1 = np.empty((len(cell1), len(p1), 2), dtype=np.float32)
    assigned_p2 = np.empty((len(cell1), len(p2), 2), dtype=np.float32)
    assigned_p1[:] = p1[None]
    assigned_p2[:] = p2[None]
    assigned_p1[~assigned] = np.nan
    assigned_p2[~assigned] = np.nan

    # find the median center and scale of assigned points in each cell
    # cell_center1 = np.nanmean(assigned_p1, axis=1)
    cell_center2 = np.nanmean(assigned_p2, axis=1)
    im1_q25, im1_q75 = np.nanquantile(assigned_p1, (0.1, 0.9), axis=1)
    im2_q25, im2_q75 = np.nanquantile(assigned_p2, (0.1, 0.9), axis=1)

    robust_std1 = (im1_q75 - im1_q25).clip(20.)
    robust_std2 = (im2_q75 - im2_q25).clip(20.)

    cell_size1 = (cell1[:, 2:4] - cell1[:, 0:2])
    cell_size2 = cell_size1 * robust_std2 / robust_std1
    cell2 = np.c_[cell_center2 - cell_size2 / 2, cell_center2 + cell_size2 / 2]

    # make sure cell bounds are valid
    cell2 = _norm_windows(cell2, H2, W2, forced_resolution=forced_resolution)

    # compute correspondence weights
    corres_weights = _weight_pixels(cell1, p1, assigned) * _weight_pixels(cell2, p2, assigned)

    # return a list of window pairs and assigned correspondences
    return cell1, cell2, corres_weights
```

```python
def greedy_selection(corres_weights, target=0.9):
    # corres_weight = (n_cell_pair, n_corres) matrix.
    # If corres_weight[c,p]>0, means that correspondence p is visible in cell pair p
    assert 0 < target <= 1
    corres_weights = corres_weights.copy()

    total = corres_weights.max(axis=0).sum()
    target *= total

    # init = empty
    res = []
    cur = np.zeros(corres_weights.shape[1])  # current selection

    while cur.sum() < target:
        # pick the nex best cell pair
        best = corres_weights.sum(axis=1).argmax()
        res.append(best)

        # update current
        cur += corres_weights[best]
        # print('appending', best, 'with score', corres_weights[best].sum(), '-->', cur.sum())

        # remove from all other views
        corres_weights = (corres_weights - corres_weights[best]).clip(min=0)

    return res
```

# Experimental results

## Training

14 datasets:Habitat [74], ARKitScenes [20], Blended MVS [112], MegaDepth [48], Static Scenes 3D [57],ScanNet++ [113], CO3D-v2 [67], Waymo [83], Map- free [5], WildRgb [2], VirtualKitti [12], Unreal4K [91], TartanAir [103] and an internal dataset. Covers indoor, outdoor, synthetic, real-world, object-centric, etc. Among them, 10 datasets have metric ground-truth. 

Specifically, we utilize off-the-shelf image retrieval and point matching algorithms to **match and verify image pairs**.

**Training**

checkpoint of DUSt3R (Vit-L encoder & Vit-B decoder). 65k equally distributed image pair per iteration.

**Correspondences**

![image.png](images/Mast3R%20Matching%20And%20Stereo%203D%20Reconstruction%201e171bdab3cf80af9ad8fe842e9b4c83/image%2016.png)

![image.png](images/Mast3R%20Matching%20And%20Stereo%203D%20Reconstruction%201e171bdab3cf80af9ad8fe842e9b4c83/image%2017.png)

![image.png](images/Mast3R%20Matching%20And%20Stereo%203D%20Reconstruction%201e171bdab3cf80af9ad8fe842e9b4c83/image%2018.png)
