# VGGT-SLAM Report

# VGGT-SLAM: Dense RGB SLAM Optimized on the SL(4) Manifold - Technical Report

[paper](https://arxiv.org/pdf/2505.12549)

[code](https://github.com/MIT-SPARK/VGGT-SLAM)

## Executive Summary

VGGT-SLAM is a novel dense RGB SLAM system that constructs consistent maps by incrementally aligning submaps created from the feed-forward scene reconstruction approach VGGT using only uncalibrated monocular cameras. The key innovation lies in recognizing that traditional similarity transform (Sim(3)) alignment is insufficient for uncalibrated camera reconstruction due to projective ambiguity, leading to the development of the first SLAM system optimized on the SL(4) manifold with 15 degrees of freedom.

## 1. Introduction and Motivation

### Problem Context

Traditional SLAM approaches rely on classical multi-view geometry constraints, data association, and bundle adjustment optimization. Recent feed-forward networks like DUSt3R have simplified this by directly producing point clouds from uncalibrated images. VGGT extends this concept to handle arbitrary numbers of frames while estimating dense point clouds, depth maps, feature tracks, and camera poses.

### Core Challenge

**VGGT** is limited in the number of frames that can be processed by GPU memory - approximately **60 frames on an NVIDIA GeForce RTX 4090** with 24 GB, making larger reconstructions requiring hundreds or thousands of frames infeasible.

### Key Insight

While conventional approaches might attempt simple Sim(3) alignment between overlapping submaps, the feed-forward nature of VGGT with uncalibrated cameras introduces a projective ambiguity, which in addition to the **Sim(3) DOF includes shear, stretch, and perspective DOF**, especially when the disparity between frames becomes small.

## 2. Technical Approach

### 2.1 Projective Ambiguity Analysis

The fundamental theoretical foundation stems from the Projective Reconstruction Theorem, which states that given a set of uncalibrated cameras with no assumption on the camera motion or scene structure, the scene can only be reconstructed up to a **15-degrees-of-freedom** projective transformation of the true geometry.

This manifests as a 4×4 homography matrix H belonging to the **Special Linear group SL(4)**, which differs from the more common 8-DOF homography used in planar image alignment tasks.

### 2.2 System Architecture

### Submap Generation

1. **Keyframe Selection**: Frames are selected as keyframes when **disparity** (estimated using **Lucas-Kanade) exceeds a threshold τ_disparity (25 pixels)** relative to the previous keyframe
2. **Submap Construction**: Each submap’s image set **I_latest** contains:
    - **Up to w**  frames
    - **Last non-loop-closure frame** from the **previous** submap (M_prior)
    - Up to **w_loop** frames for loop closures (I_loop)
    - All **keyframes**

![image.png](images/VGGT-SLAM%20Report%2022d71bdab3cf806baa3dd9ace48cbe33/image.png)

### Local Submap Alignment

For two overlapping submaps S_i and S_j with point clouds X^{S_i} and X^{S_j}, the system solves for a transformation H^i_j ∈ ℝ^{4×4} such that:

```
X^{S_i}_a = H^i_j X^{S_j}_b                     （1）
```

The optimal homography is computed by solving the homogeneous linear system:

```
	A_k h = 0                                     （2）
```

Where h contains the flattened 16 parameters of the homography matrix, Ak is the k-th pair of 3D points. The system uses **RANSAC with a 5-point solver** for robustness against incorrect depth measurements.

![image.png](images/VGGT-SLAM%20Report%2022d71bdab3cf806baa3dd9ace48cbe33/image%201.png)

VGGT camera model: simple pinhole: (cx=W/2, cy=H/2,f), **only one parameter f**, when image size is fixed.

### Loop Closure Detection

1. **Image Retrieval**: Identifies similar frames across submaps using **SALAD** descriptors, based on **L2 norm similarity**. Candidate matches are selected from the closest w_loop frames, subject to a minimum similarity threshold for Si, 1≤i≤lastest-t_interval .  **I_latest**⇒VGGT
2. **Homography Estimation**: Leverages shared frames between submaps to estimate relative homographies without requiring correspondence estimation (*between the frames in I_loop and their respective identical frames in the submap Si where they originated*)
3. SALAD is chosen for memory efficiency

### 2.3 Backend Optimization on SL(4) Manifold

The system formulates a nonlinear **factor graph optimization** problem using Maximum A Posteriori (MAP) estimation:

![image.png](images/VGGT-SLAM%20Report%2022d71bdab3cf806baa3dd9ace48cbe33/image%202.png)

![image.png](images/VGGT-SLAM%20Report%2022d71bdab3cf806baa3dd9ace48cbe33/image%203.png)

**Key technical components:**

- **Lie Group Parameterization**: Uses 15 generators G_k for the SL(4) tangent space
- **Iterative Optimization**: Employs **Levenberg-Marquardt optimizer** with proper manifold updates
- **Adjoint Representation**: Utilizes **adjoint maps for linearization** in the optimization process

## 3. Experimental Results

Evaluated on 7-Scenes and TUM RGB-D datasets against DROID-SLAM and MASt3R-SLAM baselines. worse than the **calibrated** MASt3R-SLAM.

**Key Performance**:

- **TUM RGB-D**: 0.053m average ATE (best overall vs 0.060m MASt3R-SLAM, 0.158m DROID-SLAM)
- **7-Scenes**: Competitive pose accuracy (0.067m) with best dense reconstruction metrics
- **Ablations**: Loop closures improve accuracy; larger submaps (w=32) provide better stability

## 4. Technical Innovations

1. **First SL(4) SLAM System**: Factor graph optimization on SL(4) manifold for projective ambiguity
2. **Efficient Homography Estimation**: Uses shared keyframes between submaps, avoiding correspondence estimation
3. **Hybrid Approach**: Sim(3) variant available when metric reconstruction is reliable

## 5. Limitations

- **Planar Scene Degeneracy**: Homography estimation fails in planar environments
- **Outlier Sensitivity**: High outlier ratios can cause incorrect estimates despite RANSAC
- **Computational Overhead**: SL(4) optimization more expensive than traditional SE(3)

## 6. Impact

VGGT-SLAM successfully demonstrates how classical computer vision theory (projective reconstruction) can inform and improve modern learning-based SLAM systems. By recognizing the fundamental limitations of similarity transforms for uncalibrated camera reconstruction and developing appropriate mathematical frameworks, the system achieves competitive performance while extending capabilities to scenarios previously infeasible.

The work establishes a new paradigm for SLAM optimization and provides a foundation for future research into projective geometry-aware SLAM systems. The combination of feed-forward neural reconstruction with principled geometric optimization represents a significant advance in visual SLAM capabilities.

## Technical Specifications

### Implementation Details

- **Platform**: NVIDIA GeForce RTX 4090 GPU, AMD Ryzen Threadripper 7960X CPU
- **Key Parameters**:
    - w_loop = 1
    - τ_disparity = 25 pixels
    - τ_interval = 2
    - τ_desc = 0.8
    - τ_conf = 25%
    - 300 RANSAC iterations with 0.01 threshold
