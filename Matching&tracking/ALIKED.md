# ALIKED

---

- **Full name**: ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation.
- **Paper**: https://arxiv.org/abs/2304.03608

# **Abstract:**

Image keypoints and descriptors play a crucial role in many visual measurement tasks. In recent years, deep neural networks have been widely used to improve the performance of keypoint and descriptor extraction. However, the conventional convolution operations do not provide the geometric invariance required for the descriptor. To address this issue, we propose the Sparse Deformable Descriptor Head (SDDH), which learns the deformable positions of supporting features for each keypoint and constructs deformable descriptors. Furthermore, SDDH extracts descriptors at sparse keypoints instead of a dense descriptor map, which enables efficient extraction of descriptors with strong expressiveness. In addition, we relax the neural reprojection error (NRE) loss from dense to sparse to train the extracted sparse descriptors. Experimental results show that the proposed network is both efficient and powerful in various visual measurement tasks, including image matching, 3D reconstruction, and visual relocalization.

# Architecture

![image.png](images/ALIKED%201e471bdab3cf80d496ecea3e5421f167/image.png)

![image.png](images/ALIKED%201e471bdab3cf80d496ecea3e5421f167/image%201.png)

![image.png](images/ALIKED%201e471bdab3cf80d496ecea3e5421f167/image%202.png)

## The Proposed ALIKED Framework

The ALIKED framework presents an integrated approach for keypoint detection and descriptor extraction. While the paper abstract does not delve into the specifics of the overall network architecture, it is understood that ALIKED likely follows the trend of **contemporary deep learning** methods that **jointly learn both keypoints and their descriptors within a single network**. This contrasts with earlier approaches where keypoints were detected using one method and descriptors were extracted using another. The efficiency and performance gains reported by ALIKED suggest a carefully designed architecture that the complexity required for **robust feature representation** with the need for **computational lightness**.

At the heart of the ALIKED framework lies the **Sparse Deformable Descriptor Head (SDDH)**. This novel component is responsible for learning the deformable positions of supporting features for each detected keypoint. Instead of relying on fixed convolutional grids, the SDDH can **adapt the receptive field for each keypoint** based on the local image content. This adaptability allows for the construction of descriptors that are more **robust to geometric transformations**, a known limitation of conventional convolution operations. The SDDH achieves efficiency by extracting these **deformable descriptors** only at the locations of **sparse keypoints**, rather than across an entire dense feature map. This approach is inspired by techniques used in deformable image alignment. The number of **sample locations**, denoted as **M**, within the SDDH is a **flexible** parameter, allowing the network to model a variety of geometric transformations. By operating **sparsely**, the SDDH aims to improve upon the efficiency of methods like **Deformable Convolutional Networks (DCN)**, which can introduce additional computational overhead due to the computation of dense descriptor maps.

It employs a **differentiable keypoint detection (DKD)** mechanism, enabling accurate **sub-pixel keypoint localization**. This is often achieved using a **SoftArgMax** operation, which allows for the estimation of keypoint locations with sub-pixel precision. It is mentioned that ALIKED also utilizes differentiable keypoint detection instead of non-maximum suppression (NMS). This suggests that ALIKED likely benefits from the accurate keypoint localization capabilities inherent in the ALIKE architecture.

The generation of the descriptor in ALIKED is based on the output of the SDDH. The SDDH learns the **optimal deformable positions** around each keypoint, and these learned features are then used to construct a descriptor vector. The dimensionality of this descriptor is an important factor in its expressiveness and the efficiency of subsequent matching processes. Based on information from image matching challenges, ALIKED descriptors have been reported to be **128-dimensional floating-point vectors**, occupying 512 bytes. This relatively compact descriptor size contributes to the overall lightweight nature of the ALIKED network.

# Handling Sparse Descriptors with Relaxed NRE Loss

A significant challenge in training networks that extract sparse descriptors, such as ALIKED, lies in the application of appropriate loss functions. The **Neural Reprojection Error (NRE)** loss is a powerful technique commonly used for training keypoint descriptors. The absence of a complete descriptor map means that the probability maps required by the dense NRE loss cannot be constructed.

To address this challenge, the authors of ALIKED propose an innovative solution: **relaxing the NRE loss from a dense to a sparse formulation**. ALIKED constructs **sparse probability vectors** specifically for the extracted sparse descriptors. The training objective then becomes minimizing the distance between these sparse matching probability vectors and the corresponding sparse reprojection probability vectors.  

This sparse NRE loss offers several key benefits. Firstly, it directly tackles the problem of training sparse descriptors without requiring a dense representation. Secondly, by operating on sparse data, it significantly reduces the amount of redundant computations during the network training process. This reduction in computation directly translates to substantial memory savings on the Graphics Processing Unit (GPU), which is a critical factor for enabling the training of deeper and more complex models, or for training on larger datasets.

![image.png](images/ALIKED%201e471bdab3cf80d496ecea3e5421f167/image%203.png)

![image.png](images/ALIKED%201e471bdab3cf80d496ecea3e5421f167/image%204.png)

![image.png](images/ALIKED%201e471bdab3cf80d496ecea3e5421f167/image%205.png)

![image.png](images/ALIKED%201e471bdab3cf80d496ecea3e5421f167/image%206.png)

![image.png](images/ALIKED%201e471bdab3cf80d496ecea3e5421f167/image%207.png)

![image.png](images/ALIKED%201e471bdab3cf80d496ecea3e5421f167/image%208.png)

![image.png](images/ALIKED%201e471bdab3cf80d496ecea3e5421f167/image%209.png)

### Key Features:

- **Sparse Deformable Descriptor Head (SDDH):** This is a novel component that learns deformable positions of supporting features for each detected keypoint, enabling the construction of descriptors robust to geometric transformations.
- **Efficient Descriptor Extraction:** The SDDH extracts descriptors only at sparse keypoints, rather than across a dense feature map, leading to significant computational efficiency.
- **Relaxed NRE Loss:** The paper introduces a method to relax the Neural Reprojection Error (NRE) loss from a dense to a sparse formulation to effectively train the sparse descriptors extracted by the SDDH.
- **Lightweight Design:** ALIKED is designed to be a lighter network, making it suitable for resource-constrained platforms and real-time applications.
- **High Performance in Visual Measurement Tasks:** Experimental results demonstrate that ALIKED achieves excellent performance in image matching, 3D reconstruction, and visual relocalization.
- **Differentiable Keypoint Detection (DKD):** ALIKED utilizes DKD instead of non-maximum suppression (NMS), likely for more accurate keypoint localization and to enable end-to-end training.
- **Compact Descriptors:** ALIKED uses 128-dimensional floating-point descriptors, contributing to its efficiency.
- **Rotation Augmentation:** Some versions of ALIKED, like `aliked-n16rot`, are trained with rotation augmentation to improve robustness to viewpoint changes.

---

[**LightGlue** (2023)](https://www.notion.so/LightGlue-2023-1e471bdab3cf80f8b21ffa28abf8da00?pvs=21)

---

## **Summary Table**

| Aspect | ALIKED | LightGlue |
| --- | --- | --- |
| What it does | Keypoint detection + description | Feature matching |
| Input | Image | Keypoints + Descriptors |
| Output | Keypoints + Descriptors | Matched Keypoint Pairs |
| Speed | Very fast | Very fast |
| Model size | Small | Small (compared to SuperGlue) |
| Usage | Frontend of visual pipelines | Matching step after keypoints are extracted |

---

### 🚀 **In practice**

A **modern local feature matching pipeline** might look like:

1. **Extract keypoints and descriptors** with **ALIKED**.
2. **Match them smartly** with **LightGlue**.

This combination is super popular now in **fast SLAM**, **AR**, **3D reconstruction**, and **mobile applications**.

---