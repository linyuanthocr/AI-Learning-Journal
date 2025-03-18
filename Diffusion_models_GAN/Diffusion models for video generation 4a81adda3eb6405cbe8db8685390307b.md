# Diffusion models for video generation

[Diffusion Models for Video Generation](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)

Background:

1. It has extra requirements on temporal consistency across frames in time, which naturally demands more world knowledge to be encoded into the model.
2. In comparison to text or images, it is more difficult to collect large amounts of high-quality, high-dimensional video data, let along text-video pairs.

# **Video Generation Modeling from Scratch**

## **Parameterization & Sampling Basic**

![Untitled](Diffusion%20models%20for%20video%20generation%204a81adda3eb6405cbe8db8685390307b/Untitled.png)

![Untitled](Diffusion%20models%20for%20video%20generation%204a81adda3eb6405cbe8db8685390307b/Untitled%201.png)

[arxiv.org](https://arxiv.org/pdf/2202.00512)

![Untitled](Diffusion%20models%20for%20video%20generation%204a81adda3eb6405cbe8db8685390307b/Untitled%202.png)

# Model Architecture: 3D U-Net & DiT

Similar to text-to-image diffusion models, U-net and Transformer are still two [common architecture choices](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#model-architecture). There are a series of diffusion video modeling papers from Google based on the U-net architecture and a recent Sora model from OpenAI leveraged the Transformer architecture.

**VDM** ([Ho & Salimans, et al. 2022](https://arxiv.org/abs/2204.03458)) adopts the standard diffusion model setup but with an altered architecture suitable for video modeling. It extends the [2D U-net](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#model-architecture) to work for 3D data ([Cicek et al. 2016](https://arxiv.org/abs/1606.06650)), where each feature map represents a 4D tensor of frames x height x width x channels. This 3D U-net is factorized over space and time, meaning that each layer only operates on the space or time dimension, but not both:

- Processing *Space*:
    - Each old 2D convolution layer as in the 2D U-net is extended to be space-only 3D convolution; precisely, 3x3 convolutions become 1x3x3 convolutions.
    - Each spatial attention block remains as attention over space, where the first axis (`frames`) is treated as batch dimension.
- Processing *Time*:
    - A temporal attention block is added after each spatial attention block. It performs attention over the first axis (`frames`) and treats spatial axes as the batch dimension. The [relative position](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/#relative-position-encoding) embedding is used for tracking the order of frames. The temporal attention block is important for the model to capture good temporal coherence.
    
    ![Untitled](Diffusion%20models%20for%20video%20generation%204a81adda3eb6405cbe8db8685390307b/Untitled%203.png)