# VGGSFM

https://vggsfm.github.io/

https://arxiv.org/abs/2312.04563

[https://github.com/facebookresearch/vggsfm](https://github.com/facebookresearch/vggsfm)

# **Abstract**

Structure-from-motion (SfM) is a long-standing problem in the computer vision community, which aims to reconstruct the camera poses and 3D structure of a scene from a set of **unconstrained 2D images**. Classical frameworks solve this problem in an incremental manner by detecting and matching keypoints, registering images, triangulating 3D points, and conducting bundle adjustment. Recent research efforts have predominantly revolved around harnessing the power of deep learning techniques to enhance specific elements (e.g., keypoint matching), but are still based on the original, non-differentiable pipeline. Instead, we propose a new deep SfM pipeline VGGSfM, where **each component is fully differentiable** and thus can be trained in an **end-to-end manner**. To this end, we introduce new mechanisms and simplifications. First, we build on recent advances in **deep 2D point tracking** to extract reliable pixel-accurate tracks, which eliminates the need for chaining pairwise matches. Furthermore, we **recover all cameras simultaneously** based on the image and track features instead of gradually registering cameras. Finally, we optimise the cameras and triangulate 3D points via a **differentiable bundle adjustment** layer. We attain state-of-the-art performance on three popular datasets, CO3D, IMC Phototourism, and ETH3D.

# Method

Our method extracts 2D tracks from input images, reconstructs cameras using image and track features, initializes a point cloud based on these tracks and camera parameters, and applies a bundle adjustment layer for reconstruction refinement. The whole framework is fully differentiable.

![image.png](images/VGGSFM%201e771bdab3cf8096842bef674d70a3d6/image.png)

### Problem setting

![image.png](images/VGGSFM%201e771bdab3cf8096842bef674d70a3d6/image%201.png)

### Overview

![image.png](images/VGGSFM%201e771bdab3cf8096842bef674d70a3d6/image%202.png)

## **VGGSfM: A Visual Geometry Grounded Deep SfM Pipeline**

VGGSfM redesigns the Structure from Motion (SfM) pipeline using deep learning principles while retaining a grounding in established geometric concepts. Its core philosophy is end-to-end differentiability, meaning every major stage is implemented using differentiable modules, primarily neural networks. This allows the entire system to be trained jointly by optimizing a final reconstruction objective, typically the reprojection error minimized during bundle adjustment.

### **System Overview: The Four Differentiable Stages**

The VGGSfM system comprises four main differentiable stages :

1. **Track Extraction:** Directly outputs consistent 2D point trajectories (tracks) spanning multiple views using a deep feed-forward tracking network.
2. **Camera Recovery:** Simultaneously estimates initial camera parameters (intrinsics and extrinsics) for all views using a Transformer-based network.
3. **Triangulation:** Computes the initial 3D structure (sparse point cloud) using a dedicated, learnable Transformer-based network.
4. **Bundle Adjustment (BA):** Jointly refines camera parameters and 3D point coordinates using a differentiable BA layer.
    
    ![image.png](images/VGGSFM%201e771bdab3cf8096842bef674d70a3d6/image%203.png)
    

### **Direct 2D Track Extraction Method**

Instead of traditional pairwise matching and chaining, VGGSfM employs a **deep feed-forward tracking function** based on recent video point tracking advancements.

- **Architecture:** It uses a 2D Convolutional Neural Network (CNN) backbone (inspired by PIPS ) to extract dense feature maps. Query point descriptors are correlated with feature maps across multiple resolutions (cost-volume pyramid). A **Transformer network** processes this correlation information to predict the final tracks. Each track contains 2D locations and visibility indicators.
    
    ![image.png](images/VGGSFM%201e771bdab3cf8096842bef674d70a3d6/87fe6e2b-542f-4cfd-86eb-8794ca1048b6.png)
    
    ![image.png](images/VGGSFM%201e771bdab3cf8096842bef674d70a3d6/image%204.png)
    
    ![image.png](images/VGGSFM%201e771bdab3cf8096842bef674d70a3d6/image%205.png)
    
    Note:  1. Does not assume temporal continuity. 2. Predict each track independently
    
    **Tracking Confidence**
    
    ![image.png](images/VGGSFM%201e771bdab3cf8096842bef674d70a3d6/image%206.png)
    
- **Refinement:** A **coarse-to-fine** strategy is used. Initial coarse tracks are refined using a shallower network processing higher-resolution image patches around the coarse estimates to achieve **sub-pixel accuracy**.
    1. coarsely track image points using feature maps (whole input image)
    2. we form P × P patches by cropping input images around the coarse point estimates and execute the tracking again to obtain a sub-pixel estimate. 
    3. our tracker is fully differentiable. This enables **back-propagating the gradient of the training loss L** through the whole framework to the tracker parameters.
- **Innovation:** This direct multi-view track extraction **eliminates explicit pairwise matching and chaining**, simplifying correspondence estimation and potentially reducing error propagation.

### **Simultaneous Camera Parameter Estimation Method**

VGGSfM estimates parameters for all cameras simultaneously using a deep Transformer network, replacing the **sequential registration (incremental loop)** of classical SfM.

- **Inputs:** The network uses features from input images (e.g., ResNet backbone) and extracted 2D tracks.
- **Architecture: Cross-attention** mechanisms fuse image and track features. A preliminary **relative pose estimate** (using a batched 8-point algorithm on tracks to approximate RANSAC ) is **embedded and combined with learned features**. This fused information feeds into a Transformer encoder (multiple self-attention layers). An MLP head predicts refined camera parameters (rotation, translation, focal length).
- **Refinement:** The process is iterative within the network, using the prediction from one pass as the initialization for the next.
- **Innovation:** Simultaneous, learnable recovery replaces sequential, potentially drift-prone registration, enabling joint optimization suitable for end-to-end learning.

![image.png](images/VGGSFM%201e771bdab3cf8096842bef674d70a3d6/image%207.png)

### **Transformer-Based 3D Point Triangulation Method**

VGGSfM uses a learnable Transformer network for triangulation.

- **Initialization:** A preliminary 3D point cloud is computed using a standard multi-view Direct Linear Transform (DLT) method based on initial camera estimates and 2D tracks.
- **Inputs:** The network takes track point features and positional information derived from the preliminary DLT points (e.g., coordinates encoded via positional harmonic embeddings, point-to-ray distances).
- **Architecture:** Inputs are concatenated and fed into a Transformer network (self-attention layers). An MLP head predicts the final, refined 3D coordinates.
- **Innovation:** The learnable module can potentially handle noise better than traditional methods by leveraging learned priors and integrating information via attention mechanisms.

![image.png](images/VGGSFM%201e771bdab3cf8096842bef674d70a3d6/image%208.png)

### **Differentiable Bundle Adjustment Method**

The final refinement stage uses a differentiable Bundle Adjustment (BA) layer, crucial for end-to-end training.

- **Function:** Takes initial camera and point estimates, plus 2D tracks, and minimizes the reprojection error (sum of squared distances between projected 3D points and observed 2D track locations).
- **Differentiability:** Achieved using libraries like Theseus. Theseus implements differentiable non-linear least squares, allowing gradient computation with respect to the BA layer's inputs via the implicit function theorem. This enables backpropagation through the optimization process.
- **Training vs. Inference:** A fixed number of optimization steps (e.g., 5) are used during training for efficiency, while more steps or convergence criteria (e.g., 30 steps) are used during inference.

![image.png](images/VGGSFM%201e771bdab3cf8096842bef674d70a3d6/image%209.png)

![image.png](images/VGGSFM%201e771bdab3cf8096842bef674d70a3d6/image%2010.png)

**Innovation:**

Replaces traditional non-differentiable solvers (like Ceres) with a differentiable layer, enabling gradient flow through the entire pipeline for holistic training.

### Method detials

8 degree: R→ quaternion (4), T(3), log(focal_length(1)

![image.png](images/VGGSFM%201e771bdab3cf8096842bef674d70a3d6/image%2011.png)

![image.png](images/VGGSFM%201e771bdab3cf8096842bef674d70a3d6/image%2012.png)
