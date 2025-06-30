# VGGT: Visual Geometry Grounded Transformer

[https://github.com/facebookresearch/vggt](https://github.com/facebookresearch/vggt)

https://arxiv.org/abs/2503.11651

![image.png](images/VGGT%20Visual%20Geometry%20Grounded%20Transformer%201ba71bdab3cf80c88bf9eee2a0d5313f/image.png)

## Contributions

1. VGGT, a large feedforward transformer that directly infers all key 3D attributes of a scene, including camera parameters, point maps, depth maps, and 3D point tracks, from one, a few, or hundreds of its views.
2. simple, efficient, reconstruction images in under one second. outperform opimization based methods 
3. state of art results in multiple 3D task, directly usable
4. combined with BA post-processing, VGGT state of art results across the board

![image.png](images/VGGT%20Visual%20Geometry%20Grounded%20Transformer%201ba71bdab3cf80c88bf9eee2a0d5313f/image%201.png)

train dataset：public dataset+3D anonotation

## Method

### Problem defintion

![image.png](images/VGGT%20Visual%20Geometry%20Grounded%20Transformer%201ba71bdab3cf80c88bf9eee2a0d5313f/image%202.png)

the **point maps** are **viewpoint invariant**, meaning that the 3D points Pi are defined in the coordinate system of the first camera g1, which we take as the world reference frame

camera parameters:

![image.png](images/VGGT%20Visual%20Geometry%20Grounded%20Transformer%201ba71bdab3cf80c88bf9eee2a0d5313f/image%203.png)

depth & point map, no special

keypoint tracking (two networks trained jointly end to end, f: encoder, T: tracking):

![image.png](images/VGGT%20Visual%20Geometry%20Grounded%20Transformer%201ba71bdab3cf80c88bf9eee2a0d5313f/image%204.png)

Order prediction: permutation **equivariant** for all but the first frame.

Over-complete Predictions: tasking VGGT with explicitly predicting all afore-mentioned quantities during training. brings substantial performance gains, even when these are related by closed-form relationships. 

### Feature backbone

a simple architecture with **minimal 3D inductive biases**, letting the model learn from ample quantities of 3D-annotated data. 

input image I is initially **patchified** into a set of **K tokens**  tI ∈ RK×C through **DINO** [78]. The combined set of image tokens from **all frames**, is subsequently processed through the main network structure, **alternating frame-wise and global self-attention layers**. 

L = 24, our architecture does not employ any cross-attention layers, only self-attention ones.

### Prediction head

![image.png](images/VGGT%20Visual%20Geometry%20Grounded%20Transformer%201ba71bdab3cf80c88bf9eee2a0d5313f/image%205.png)

![image.png](images/VGGT%20Visual%20Geometry%20Grounded%20Transformer%201ba71bdab3cf80c88bf9eee2a0d5313f/image%206.png)

![image.png](images/VGGT%20Visual%20Geometry%20Grounded%20Transformer%201ba71bdab3cf80c88bf9eee2a0d5313f/image%207.png)

### **Camera prediction  (intrinsics and extrinsics)**

![image.png](images/VGGT%20Visual%20Geometry%20Grounded%20Transformer%201ba71bdab3cf80c88bf9eee2a0d5313f/image%208.png)

### Dense predictions

Depth estimation

![image.png](images/VGGT%20Visual%20Geometry%20Grounded%20Transformer%201ba71bdab3cf80c88bf9eee2a0d5313f/image%209.png)

Head part is like:

![image.png](images/VGGT%20Visual%20Geometry%20Grounded%20Transformer%201ba71bdab3cf80c88bf9eee2a0d5313f/image%2010.png)

![image.png](images/VGGT%20Visual%20Geometry%20Grounded%20Transformer%201ba71bdab3cf80c88bf9eee2a0d5313f/image%2011.png)

**DPT head is the fusion+head part of the original DPT, only add different layers output+upsample+conv (no ressamples module from original DPT)**

### How the Tracking Head Works

**Dense Feature Generation:**

- The output image tokens  $t̂^I_i$ are used to predict the dense outputs, i.e., the depth maps Di, point maps Pi, and tracking features Ti
- $t̂^I_i$ are first converted to dense feature maps $Fi ∈ R^(C''×H×W)$ with a DPT layer
- The DPT head also outputs dense features $Ti ∈ R^(C×H×W)$, which serve as input to the tracking head

**Tracking Module Architecture:**

- We use the **CoTracker2** architecture, which takes the dense tracking features Ti as input
- Given a query point yj in a query image Iq (during training, we always set q = 1, but any other image can be potentially used as a query), the tracking head T predicts the set of 2D points $T((yj)^M_{j=1}, (Ti)^N_{i=1}) = ((ŷj,i)^N_{i=1})^M_{j=1}$

**Feature Correlation Process:**

1. The feature map Tq of the query image is first bilinearly sampled at the query point yj to obtain its feature
2. This feature is then correlated with all other feature maps Ti, i ≠ q to obtain a set of correlation maps
3. These maps are then processed by self-attention layers to predict the final 2D points ŷi, which are all in correspondence with yj

**Key Design Choice:**

- Similar to VGGSfM, our tracker does not assume any temporal ordering of the input frames and, hence, can be applied to any set of input images, not just videos

## Training (Section 3.4)
![image.png](https://github.com/linyuanthocr/AI-Learning-Journal/blob/main/3D%20reconstruction/images/image.png)

**Training Setup:**

- 1.2 billion parameters total
- AdamW optimizer for 160K iterations
- Peak learning rate of 0.0002 with warmup of 8K iterations
- Randomly sample 2–24 frames from a random training scene
- Training runs on 64 A100 GPUs over nine days

**Key Insight:** Unlike [129], we do not apply such normalization to the predictions output by the transformer; instead, we force it to learn the normalization from the training data

## Reference

[**DUSt3R: Geometric 3D Vision Made Easy**](https://www.notion.so/DUSt3R-Geometric-3D-Vision-Made-Easy-1ba71bdab3cf80a08e7afbedcb4a1605?pvs=21)

https://arxiv.org/abs/2312.14132

![image.png](images/VGGT%20Visual%20Geometry%20Grounded%20Transformer%201ba71bdab3cf80c88bf9eee2a0d5313f/image%2012.png)

![image.png](images/VGGT%20Visual%20Geometry%20Grounded%20Transformer%201ba71bdab3cf80c88bf9eee2a0d5313f/image%2013.png)

[https://github.com/naver/dust3r](https://github.com/naver/dust3r)

**Grounding Image Matching in 3D with MASt3R**

https://arxiv.org/abs/2406.09756

[https://github.com/naver/mast3r](https://github.com/naver/mast3r)

![image.png](images/VGGT%20Visual%20Geometry%20Grounded%20Transformer%201ba71bdab3cf80c88bf9eee2a0d5313f/image%2014.png)

![image.png](images/VGGT%20Visual%20Geometry%20Grounded%20Transformer%201ba71bdab3cf80c88bf9eee2a0d5313f/image%2015.png)
