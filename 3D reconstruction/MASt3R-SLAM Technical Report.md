# MASt3R-SLAM Technical Report

[MASt3R-SLAM paper](https://arxiv.org/abs/2412.12392)

[MASt3R-SLAM project](https://edexheim.github.io/mast3r-slam/)

## Executive Summary

MASt3R-SLAM represents a significant advancement in real-time monocular dense SLAM technology by leveraging learned 3D reconstruction priors from the MASt3R two-view 3D reconstruction network. Developed by researchers at Imperial College London (Riku Murai, Eric Dexheimer, and Andrew J. Davison), this system achieves state-of-the-art performance in dense geometry reconstruction while maintaining real-time operation at 15 FPS without requiring camera calibration.

**Key Innovation**: First real-time SLAM system to fully integrate two-view 3D reconstruction priors into an incremental mapping framework, enabling globally consistent dense reconstruction from uncalibrated monocular video.

## Technical Background

### Foundation: MASt3R Prior

![image.png](images/MASt3R-SLAM%20Technical%20Report%2022d71bdab3cf801b9d7ef5bc56190639/image.png)

MASt3R (Multi-view Stereo 3D Reconstruction) is a foundational model that outputs dense pointmaps and per-pixel features from two input images in a common coordinate frame. Unlike traditional SLAM approaches that solve camera poses, 3D structure, and correspondences separately, MASt3R provides a unified solution by:

- Generating dense 3D pointmaps directly from image pairs
- Providing per-pixel confidence scores and feature descriptors
- Outputting results in a shared coordinate frame without requiring camera calibration
- Enabling robust matching across extreme viewpoint changes

### Problem Statement

Traditional dense SLAM systems face several limitations:
- Dependency on parametric camera models and calibration
- Computational bottlenecks in dense matching and optimization
- Poor performance with time-varying camera parameters
- Limited robustness in challenging real-world scenarios

MASt3R-SLAM addresses these challenges by building a complete SLAM pipeline around the strong 3D priors provided by MASt3R.

## System Architecture

![image.png](images/MASt3R-SLAM%20Technical%20Report%2022d71bdab3cf801b9d7ef5bc56190639/image%201.png)

### Pipeline Overview

The MASt3R-SLAM system consists of five main components:

1. **MASt3R Prediction and Pointmap Matching**
2. **Front-End Tracking and Local Fusion**
3. **Keyframe Graph Construction and Loop Closure**
4. **Second-Order Global Optimization**
5. **Relocalization and Camera Handling**

### 1. Pointmap Matching with Iterative Projective Matching (IPM)

**Challenge**: Direct dense matching between pointmaps is computationally expensive ( MASt3R taking ~2 seconds per frame).

**Solution**: Iterative Projective Matching technique that:
- Normalizes pointmaps into **ray representations** using a **generic central camera model**
- Performs **massively parallel matching** by **minimizing angular error between rays**
- Uses custom CUDA kernels for GPU acceleration
- Achieves matching in just 2ms per frame (40x speedup)

![image.png](images/MASt3R-SLAM%20Technical%20Report%2022d71bdab3cf801b9d7ef5bc56190639/image%202.png)

**Technical Details**:
- Each pixel in the reference frame (frame i) is normalized to **a unit ray** from the camera center

![image.png](images/MASt3R-SLAM%20Technical%20Report%2022d71bdab3cf801b9d7ef5bc56190639/image%203.png)

- Angular difference between rays is minimized using Levenberg-Marquardt optimization

![image.png](images/MASt3R-SLAM%20Technical%20Report%2022d71bdab3cf801b9d7ef5bc56190639/436e84b3-c599-4ecc-9637-a01c3623c81c.png)

- Feature refinement step improves accuracy using MASt3R’s **per-pixel descriptors**
- Unbiased matching relies purely on MASt3R outputs, not pose estimates

### 2. Camera Tracking and Local Fusion

**Generic Central Camera Model**: Instead of assuming a fixed parametric camera model (e.g., pinhole), the system:
- Normalizes pointmaps into rays passing through a unique camera center
- Enables SLAM with time-varying camera models (e.g., dynamic zoom changes up to 5x)
- Removes dependency on camera calibration (3D sapce alignment)

**Ray-Based Error Formulation**: Unlike traditional methods using 3D point errors, MASt3R-SLAM uses:

![image.png](images/MASt3R-SLAM%20Technical%20Report%2022d71bdab3cf801b9d7ef5bc56190639/image%204.png)

- Angular differences between rays for robust tracking (current frame vs the last keyframe)
- Huber norm to downweight outliers
- More stable optimization in the presence of noisy pointmap predictions

![image.png](images/MASt3R-SLAM%20Technical%20Report%2022d71bdab3cf801b9d7ef5bc56190639/9ac0be5a-5b65-4847-a548-3067f9a1744c.png)

**Local Fusion**: Geometric information is incrementally fused into keyframes using:
- **Weighted running average** for pointmap refinement
- Confidence-based weighting to **reduce noise**
- Preservation of fine geometric detail while improving **consistency**

![image.png](images/MASt3R-SLAM%20Technical%20Report%2022d71bdab3cf801b9d7ef5bc56190639/image%205.png)

### 3. Graph Construction and Loop Closure

**Feature-Based Retrieval**:
- Uses **ASMK** (Aggregated Selective Match Kernels) for efficient loop detection
- MASt3R features encoded into a retrieval database (incrementally build)
- Candidates decoded by MASt3R for validation

**Graph Structure**:
- Keyframe-based representation with Sim(3) pose relationships (**7N*7N** sim3 to be optimized)
- Edges added when sufficient matches are found between keyframes
- Supports both tracking edges and loop closure edges

### 4. Second-Order Global Optimization

**Gauss-Newton Optimization**:
- Replaces gradient descent for faster convergence
- Enables efficient large-scale pose and geometry updates
- Maintains global consistency in real-time

![image.png](images/MASt3R-SLAM%20Technical%20Report%2022d71bdab3cf801b9d7ef5bc56190639/image%206.png)

**Sparse Cholesky Decomposition**:
- Efficient computation for large-scale optimization problems
- Minimizes computational overhead while ensuring accuracy

**Joint Pose-Geometry Optimization**:
- Simultaneously optimizes camera poses and dense geometry
- Maintains coherence of original MASt3R predictions
- Scales efficiently with scene size

### 5. Relocalization

**Robust Recovery**: When tracking fails:
- Queries feature database for potential matches
- Uses MASt3R predictions for pose recovery
- Seamlessly resumes operation without manual intervention

## Key Technical Innovations

### Ray-Based Representation

- **Innovation**: Normalizing pointmaps to ray representations enables generic camera model support
- **Benefit**: Works with time-varying intrinsics, distortion, and zoom changes
- **Implementation**: All rays pass through a single camera center, removing parametric model constraints

### Efficient Dense Matching

- **Innovation**: Iterative Projective Matching with GPU acceleration
- **Benefit**: 40x speedup compared to naive dense matching (2ms vs 2000ms)
- **Implementation**: Custom CUDA kernels for parallel angular error minimization

### Confidence-Weighted Fusion

- **Innovation**: Uses MASt3R confidence scores for weighted pointmap fusion
- **Benefit**: Reduces noise while preserving accurate geometric details
- **Implementation**: Running weighted average with confidence-based weights

### Real-Time Global Optimization

- **Innovation**: Second-order optimization with sparse linear algebra
- **Benefit**: Global consistency without offline post-processing
- **Implementation**: Gauss-Newton with sparse Cholesky factorization

## Performance Evaluation

### Runtime Performance

- **Frame Rate**: 15 FPS on RTX 4090 GPU
- **Tracking**: >20 FPS for pose estimation
- **Matching**: 2ms per frame for dense correspondence
- **Network**: 64% of total runtime (encoder/decoder)

### Comparative Analysis

**vs. DROID-SLAM**:
- Better dense geometry reconstruction
- Superior performance in extreme conditions (lighting changes, dynamic objects)
- Comparable trajectory accuracy
- No camera calibration required

**vs. Spann3R**:
- More robust to non-object-centric scenes
- Better handling of larger scenes through keyframing
- Real-time global optimization capability

**vs. Traditional SLAM (ORB-SLAM)**:
- Dense reconstruction capability
- Robustness to texture-poor environments
- No feature extraction/matching required

## Limitations and Future Work

### Current Limitations

1. **Network Dependency**: MASt3R network remains the computational bottleneck
2. **Calibrated Performance**: Some accuracy benefits when intrinsics are known
3. **Pointmap Quality**: System performance depends on MASt3R prediction quality
4. **Global Geometry Refinement**: Currently limited refinement of all geometry in global optimization

### Future Research Directions

1. **Lightweight Networks**: Developing more efficient 3D reconstruction priors
2. **Global Pointmap Consistency**: Methods to refine all geometry while maintaining coherence
3. **Multi-Camera Systems**: Extending to stereo and multi-camera configurations
4. **Dynamic Scenes**: Better handling of moving objects and temporal consistency

## Technical Specifications

### Hardware Requirements

- **Recommended**: RTX 4090 GPU with i9-12th gen CPU
- **Memory**: Sufficient for pointmap storage and network inference
- **Real-time performance**: Achievable on consumer-grade hardware

### Software Dependencies

- **MASt3R**: Two-view 3D reconstruction network
- **PyTorch**: Deep learning framework
- **CUDA**: GPU acceleration for matching kernels
- **OpenCV**: Computer vision utilities

### Installation

[https://github.com/rmurai0610/MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM)

## Conclusion

MASt3R-SLAM represents a paradigm shift in dense SLAM by successfully integrating learned 3D priors into a real-time incremental mapping system. The key innovations in efficient pointmap matching, **ray-based representations**, and real-time global optimization enable robust performance across diverse scenarios without requiring camera calibration.

The system’s ability to **handle time-varying camera parameters**, achieve state-of-the-art dense reconstruction quality, and maintain real-time performance makes it a significant contribution to the SLAM community and a practical solution for real-world applications.

**Publication**: Accepted at CVPR 2025

**Authors**: Riku Murai, Eric Dexheimer, Andrew J. Davison (Imperial College London)

**Code**: Available at https://github.com/rmurai0610/MASt3R-SLAM

**Project Page**: https://edexheim.github.io/mast3r-slam/
