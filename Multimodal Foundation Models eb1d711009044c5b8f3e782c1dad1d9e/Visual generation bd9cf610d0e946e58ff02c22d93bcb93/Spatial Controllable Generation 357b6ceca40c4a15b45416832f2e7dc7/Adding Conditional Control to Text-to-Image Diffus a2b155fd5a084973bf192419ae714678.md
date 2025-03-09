# Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)

# ControlNet

[](https://arxiv.org/pdf/2302.05543.pdf)

[ControlNet](https://huggingface.co/docs/diffusers/using-diffusers/controlnet)

![Untitled](Adding%20Conditional%20Control%20to%20Text-to-Image%20Diffus%20a2b155fd5a084973bf192419ae714678/Untitled.png)

![Untitled](Adding%20Conditional%20Control%20to%20Text-to-Image%20Diffus%20a2b155fd5a084973bf192419ae714678/Untitled%201.png)

![Untitled](Adding%20Conditional%20Control%20to%20Text-to-Image%20Diffus%20a2b155fd5a084973bf192419ae714678/Untitled%202.png)

In this way, the zero convolutions become an unique type of connection layer that progressively grow from zeros to optimized parameters in a learned way.

# ControlNet in Image Diffusion Model

We use the Stable Diffusion [[44]](https://arxiv.org/pdf/2112.10752.pdf) as an example to introduce the method to use ControlNet to control a large diffusion model with task-specific conditions.

**Stable Diffusion** is a large text-to-image diffusion model trained on billions of images. The model is essentially an U-net with an encoder, a middle block, and a skip-connected decoder. Both the encoder and decoder have 12 blocks, and the full model has 25 blocks (including the middle block). In those blocks, 8 blocks are down-sampling or up-sampling convolution layers, 17 blocks are main blocks that each contains four resnet layers and **two Vision Transformers (ViTs)**. Each Vit contains several cross-attention and/or self-attention mechanisms. The texts are encoded by OpenAI CLIP, and diffusion time steps are encoded by positional encoding

[Latent Diffusion Model](https://www.notion.so/Latent-Diffusion-Model-006e610a130d46a4ad92efea6eeda78d?pvs=21) 

![Untitled](Adding%20Conditional%20Control%20to%20Text-to-Image%20Diffus%20a2b155fd5a084973bf192419ae714678/Untitled%203.png)

[](https://github.com/lllyasviel/ControlNet/blob/main/ldm/modules/diffusionmodules/model.py#L106)