# DUSt3R: Geometric 3D Vision Made Easy

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image.png)

# Introduction

**Dust3R** addresses **dense and unconstrained stereo 3D reconstruction** from arbitrary image collections—operating without any prior knowledge of **camera intrinsics or extrinsics**. It introduces a **simple yet effective global alignment strategy** that registers multiple views by expressing all pairwise **pointmaps** in a **shared reference frame**.

The core architecture is a **Transformer-based encoder-decoder**, built on **pre-trained models**, that jointly processes the **scene** and the **input images**. Given a set of **unconstrained images** as input, Dust3R outputs **dense depth maps**, **pixel-wise correspondences**, and both **relative** and **absolute camera poses**, enabling the reconstruction of a consistent **3D model**.

Dust3R is trained in a **fully-supervised** manner using a simple regression loss. It leverages large-scale datasets with **ground-truth annotations**, either synthetically generated or derived from SfM pipelines. Notably, during inference, **no geometric constraints or priors** are required—making it flexible and broadly applicable.

# Method

### Pointmap

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%201.png)

### Cameras and scene

Note: Depth image is represented with depth (not normalized invert depth)

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%202.png)

## Overview

### Inputs and outputs

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%203.png)

### **Network architecture**

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%204.png)

- **Siamese encoding** enables consistent feature extraction. （ViT）

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%205.png)

- **Generic transformer decorder**: self attention(with token+img token)+ **cross attention** + MLP
- **Cross-attention** allows views to communicate, aligning their 3D predictions.
- **Pointmaps** are generated per view but aligned through decoder interaction. (aligned to camera 1 coordinate)

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%206.png)

### **Pointmap Output and Architectural Design**

- The predicted **pointmaps** are only accurate **up to an unknown scale**.
- The architecture **does not enforce any explicit geometric constraints** (e.g., no pinhole camera model), so the pointmaps may **not strictly follow real-world camera geometry**.
- The model **learns geometrically consistent pointmaps**.
- This architecture enables the use of **powerful pretrained models**

## Training Objective

**3D Regression loss.**

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%207.png)

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%208.png)

**Confidence-aware loss.**

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%209.png)

## Downstream Applications

### Point matching

nearest neighbor (NN) search in the 3D pointmap space. Reciprocal (**Mutual)** correspondences.

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%2010.png)

### Recovering intrinsics

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%2011.png)

### Relative pose estimation

1. 2D match + camera intrinsic + essential matrix estimation
2. Procrustes alignment (close form, sensitive to noise and outliers)
    
    ![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%2012.png)
    
3. **Ransac+PNP**

### Absolute pose estimation

get the relative pose between IQ and IB as described previously. Then, we convert this pose to **world coordinate by scaling** it appropriately, according to the scale between XB,B and the ground-truth pointmap for IB.

## Global alignment

post processing to a joint 3D space with **aligned 3D point-clouds** and their corresponding **pixel-to-3D mapping**

 **pairwise graph:** all pairs ⇒ image retrival method (AP-GeM [95]), filter low average confidence pairs. (nodes: images, edge：images share visual content)

**Global optimization**：

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%2013.png)

### Recovering camera parameters

enforcing a standard camera pinhole model

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/a0cd9cc0-6522-458e-8f75-4dc40179a621.png)

*why not BA？ too long to do optimization*

# Experiments

### Find image pairs

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%2014.png)

### Train details

1. first, 224*224, then larger 512 images. randomly select image aspect ratios (crop→ largest dim to 512)
2. standard image augmentation
3. Network: **Vit-large for the encoder, Vit-base the decoder and a DPT head**.

###

# Run on Runpod
[runpod+dust3r](runpod+dust3r.md)
