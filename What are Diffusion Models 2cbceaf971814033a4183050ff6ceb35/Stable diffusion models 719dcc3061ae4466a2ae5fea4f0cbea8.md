# Stable diffusion models

# D**reamBooth**

[DreamBooth](https://huggingface.co/docs/diffusers/training/dreambooth)

[Dreambooth 官方炼丹教程万字详解-Epochs\\Batch size\\学习率 等超参数调优 （一）](https://mp.weixin.qq.com/s/8ECZ5xaUF20AqMU3jb2Zqg)

## Train on Linux

[Training Stable Diffusion with Dreambooth using Diffusers](https://huggingface.co/blog/dreambooth)

Easy to over-fit, should be smaller batch-size(2-4), very low learning rate: 2e-6, relatively longer step(800 for faces), maybe i need dreambooth + LORA

### **Summary of Initial Results**

To get good results training Stable Diffusion with Dreambooth, it's important to tune the learning rate and training steps for your dataset.

- High learning rates and too many training steps will lead to overfitting. The model will mostly generate images from your training data, no matter what prompt is used.
- Low learning rates and too few steps will lead to underfitting: the model will not be able to generate the concept we were trying to incorporate.

Faces are harder to train. In our experiments, a learning rate of `2e-6` with `400` training steps works well for objects but faces required `1e-6` (or `2e-6`) with ~1200 steps.

Image quality degrades a lot if the model overfits, and this happens if:

- The learning rate is too high.
- We run too many training steps.

The original Dreambooth paper describes a method to fine-tune the UNet component of the model but keeps the text encoder frozen. However, we observed that **fine-tuning the encoder produces better results.** Fine-tuning the text encoder produces the best results, especially with faces. It generates more realistic images, **it's less prone to overfitting and it also achieves better prompt interpretability, being able to handle more complex prompts**.

[Generate AI artwork of your Face with AI](https://medium.com/@gollulikithraj/generate-ai-artwork-of-your-face-with-ai-bee8660f09b4)

[How to Train Stable Diffusion AI with Your Face to Create Art Using DreamBooth - TechPP](https://techpp.com/2022/10/10/how-to-train-stable-diffusion-ai-dreambooth/)

# **Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference**

[Paper page - Latent Consistency Models: Synthesizing High-Resolution Images with   Few-Step Inference](https://huggingface.co/papers/2310.04378)

[](https://arxiv.org/pdf/2310.04378.pdf)

# **LoRA: Low-Rank Adaptation of Large Language Models**

[Paper page - LoRA: Low-Rank Adaptation of Large Language Models](https://huggingface.co/papers/2106.09685)

[](https://arxiv.org/pdf/2106.09685.pdf)

# **Custom Diffusion**

[Custom Diffusion](https://huggingface.co/docs/diffusers/training/custom_diffusion)

# **Multi-Concept Customization of Text-to-Image Diffusion**

[](https://arxiv.org/pdf/2212.04488.pdf)