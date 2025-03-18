# DPT：Vision Transformers for Dense Prediction

[https://github.com/isl-org/DPT](https://github.com/isl-org/DPT)

https://huggingface.co/docs/transformers/model_doc/dpt

Dense vision transformers， an architecture that leverages vision transformers in place of **convolutional networks as a backbone** for dense prediction tasks. We **assemble tokens** from **various stages** of the vision transformer into image-like representations at various resolutions and progressively combine them into **full-resolution predictions** using a **convolutional decoder**. 

![image.png](images/DPT%EF%BC%9AVision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image.png)

## Transformer encoder

DPT三种结构：

1. 原始Vit-base, 原始vit输入+12层 transformer layers， D=768
2. 原始Vit-large，原始vit输入+24层 transformer layers， D=1024
3. hybrid resnet50+12层transformer layers

pitchsize 16*16

## Convolutional decoder

three stage

![image.png](images/DPT%EF%BC%9AVision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%201.png)

### **Read**

![image.png](images/DPT%EF%BC%9AVision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%202.png)

3 operations：

![image.png](images/DPT%EF%BC%9AVision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%203.png)

### **Concatenate**

place each token acoording to the position of the initial patch in the image

![image.png](images/DPT%EF%BC%9AVision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%204.png)

### Resample

![image.png](images/DPT%EF%BC%9AVision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%205.png)

we **reassemble features at four different stages** and four differ- ent resolutions. We

combine the extracted feature maps from consecutive stages **using RefineNet-based feature fusion block,** progressively upsample the representation by a factor of two in each fusion stage.

![image.png](images/DPT%EF%BC%9AVision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%206.png)

### Handling varying image size

the **position embedding** has a dependency on the image size as it encodes the locations of the patches in the input image. We follow the approach proposed in [11] and **linearly interpolate the position embeddings** to the appropriate size. Note that this can be done on the fly for every image. 

### Depth estimation

representations of depth are **unified into a common representation** and that common ambiguities (such as **scale ambiguity**) are appropriately handled in the **training loss** [30].

![image.png](images/DPT%EF%BC%9AVision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%207.png)

a monocular depth prediction network using a scale- and shift-invariant trimmed loss that operates on an **inverse depth representation**, together with the **gradient-matching loss** proposed in [22].

1. **scale- and shift-invariant trimmed loss**

![image.png](images/DPT%EF%BC%9AVision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%208.png)

![image.png](images/DPT%EF%BC%9AVision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%209.png)

![image.png](images/DPT%EF%BC%9AVision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%2010.png)

 **b.  gradient-matching loss**

![image.png](images/DPT%EF%BC%9AVision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%2011.png)

![image.png](images/DPT%EF%BC%9AVision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%2012.png)

![image.png](images/DPT%EF%BC%9AVision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%2013.png)

https://arxiv.org/abs/1907.01341

https://arxiv.org/abs/1804.00607
