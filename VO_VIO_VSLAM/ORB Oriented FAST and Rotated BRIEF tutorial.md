# ORB - Oriented FAST and Rotated BRIEF tutorial

# ORB Feature Detection and Description Tutorial

## 1. Introduction to ORB

ORB (Oriented FAST and Rotated BRIEF) is a fast and efficient feature detection and description algorithm that combines the strengths of FAST keypoint detection and BRIEF descriptor computation.

### Key Features of ORB:

- **Fast computation**: Significantly faster than SIFT and SURF
- **Rotation invariant**: Enhanced with orientation information
- **Scale invariant**: Uses image pyramids for multi-scale detection
- **Binary descriptors**: Efficient matching using Hamming distance

## 2. Basic Concepts

### Feature Points Fundamentals

Feature points are distinctive locations in an image that remain consistent even when the camera viewpoint changes slightly. They consist of two main components:

1. **Keypoints (Key-point)**: The location of the feature in the image, sometimes including orientation and scale information
2. **Descriptors**: Usually a vector that describes the pixel information around the keypoint, following the principle that “visually similar features should have similar descriptors”

## 3. FAST Feature Detection (ORB’s Foundation)

### 3.1 FAST Algorithm Principle

FAST (Features from Accelerated Segment Test) uses a circular template with radius 3.4 pixels and 16 surrounding pixels to identify feature points.

![image.png](ORB%20-%20Oriented%20FAST%20and%20Rotated%20BRIEF%20tutorial%2024871bdab3cf80fca131c2f08cb4b25a/image.png)

**Detection Method:**
- For each pixel p, examine 16 pixels in a circle around it
- Compare intensity of each pixel with center pixel p ± threshold t
- If n consecutive pixels (typically n=12) are all brighter or all darker than the center, it’s a corner

**Pixel Classification:**

```
S_{p-x} = {
  d  if I_{p-x} ≤ I_p - t     (darker)
  s  if I_p - t < I_{p-x} < I_p + t   (similar)
  b  if I_p + t ≤ I_{p-x}     (brighter)
}
```

### 3.2 Speed Optimization

For faster detection, check only 4 pixels initially (positions 1, 5, 9, 13):
- If at least 3 of these 4 pixels differ significantly from center, proceed with full 16-pixel test
- Otherwise, skip this pixel

### 3.3 Corner Response Function

 non-maximum suppression, filter out the lower V

$$
V = max(∑_{x∈S_{bright}} |I_{p-x} - I_p| - t, ∑_{x∈S_{dark}} |I_{p-x} - I_p| - t)
$$

## 4. ORB’s Improvements to FAST

### 4.1 Scale Invariance through Image Pyramids

ORB addresses FAST’s lack of scale invariance by:
- Creating an image pyramid with scale factor (typically 1.2)
- Setting number of pyramid levels (typically 8)
- Detecting features at each scale level
- Scaled images: `$I' = I/scaleFactor^k$` where k = 1,2,…,nlevels

### 4.2 Orientation Computation

Unlike original FAST, ORB computes the main orientation of feature points to enable rotation invariance for the subsequent BRIEF descriptor.

**Centroid Calculation:**

$$
M₀₀ = ∑∑I(x,y)\\
M₁₀ = ∑∑xI(x,y)\\M₀₁ = ∑∑yI(x,y) \\
Qₓ = M₁₀/M₀₀\\ Qᵧ = M₀₁/M₀₀
$$

The orientation θ is calculated as:

$$
θ = arctan2(Qᵧ, Qₓ)
$$

## 5. BRIEF Descriptor (ORB’s Foundation)

### 5.1 BRIEF Algorithm Steps

BRIEF (Binary Robust Independent Elementary Features) creates binary descriptors:

1. **Preprocessing**: Apply Gaussian filtering (σ=2, 9×9 window) to reduce noise
2. **Random Sampling**: In an S×S neighborhood around the feature point, randomly select N pairs of points （S=31, N=256）
3. **Binary Test**: For each pair (x,y):
    
    ```
    τ(p;x,y) = {
      1  if p(x) < p(y)
      0  otherwise
    }
    ```
    
4. **Descriptor Creation**: Combine N binary tests to form an N-bit binary string

### 5.2 Sampling Patterns

Five different sampling methods (Method 2 is preferred):
1. Uniform distribution in [-S/2, S/2]
2. **Gaussian distribution** with isotropic sampling
3. Two-step Gaussian sampling
4. Discrete polar coordinate sampling
5. Fixed center with polar coordinate sampling

![image.png](ORB%20-%20Oriented%20FAST%20and%20Rotated%20BRIEF%20tutorial%2024871bdab3cf80fca131c2f08cb4b25a/image%201.png)

### 5.3 Feature Matching

- Use **Hamming distance** for matching binary descriptors
- Pairs with <128 matching bits (out of 256) are rejected
- **Select pairs with maximum bit agreement between images**

## 6. ORB’s Improvements to BRIEF

### 6.1 Rotation Invariance Enhancement

**Problem with Original BRIEF:**
- Uses fixed coordinate system (horizontal X-axis, vertical Y-axis)
- Same sampling pattern yields different descriptors when image **rotates**

**ORB’s Solution:**
- Establishes coordinate system with feature point as origin
- Uses line from feature point to region centroid as X-axis
- Ensures consistent point pair selection regardless of image rotation

### 6.2 Steered BRIEF (rBRIEF)

ORB rotates the BRIEF sampling pattern according to the keypoint orientation:

$$
S_θ = R_θ · S
$$

Where R_θ is the rotation matrix for angle θ.

## 7. Complete ORB Algorithm Pipeline

### Step 1: Multi-Scale FAST Detection

1. Create image pyramid
2. Apply FAST corner detection at each scale
3. Apply non-maximum suppression

### Step 2: Orientation Assignment

1. Compute intensity centroid for each keypoint
2. Calculate orientation angle θ
3. Assign orientation to keypoint

### Step 3: BRIEF Descriptor

1. **Rotate sampling pattern** by keypoint orientation
2. Perform binary tests on rotated pattern
3. Generate 256-bit binary descriptor

### Step 4: Feature Matching

1. Compute **Hamming distance** between descriptors
2. Apply ratio test or threshold-based filtering
3. Perform geometric verification if needed

## 8. Advantages and Applications

### Advantages:

- **Speed**: Much faster than SIFT/SURF (up to 100x faster)
- **Memory efficient**: Binary descriptors require less storage
- **Rotation invariant**: Handles image rotations well
- **Scale awareness**: Multi-scale detection capability
- **Real-time capable**: Suitable for mobile and embedded applications

### Limitations:

- Less robust to illumination changes compared to SIFT
- May struggle with significant scale variations
- Performance degrades with severe perspective distortions

## 9. Code

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('simple.jpg', cv.IMREAD_GRAYSCALE)

# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()
```

## 10. Conclusion

ORB represents an excellent balance between computational efficiency and feature quality, making it ideal for real-time computer vision applications. By combining the speed of FAST detection with the robustness of BRIEF descriptors, and adding rotation and scale invariance, ORB has become a cornerstone algorithm in modern computer vision systems.

The algorithm’s binary nature makes it particularly well-suited for resource-constrained environments while maintaining sufficient discriminative power for most practical applications.