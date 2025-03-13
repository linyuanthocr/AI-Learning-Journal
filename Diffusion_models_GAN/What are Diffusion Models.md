# What are Diffusion Models?

[What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

Diffusion models are inspired by non-equilibrium thermodynamics. They define a **Markov chain** of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise. Unlike VAE or flow models, diffusion models are learned with **a fixed procedure and the latent variable has high dimensionality (same as the original data).**

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled.png)

# What are diffusion models?

## **Forward diffusion process**

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%201.png)

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%202.png)

### **Connection with stochastic gradient Langevin dynamics**

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%203.png)

## **Reverse diffusion process**

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%204.png)

learn a model **$p_\theta(x_{t-1}|x_t)$** to approximate these conditional probabilities **$q(x_{t-1}|x_t)$** in order to **run the *reverse diffusion process*** 

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%205.png)

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%206.png)

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%207.png)

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%208.png)

To convert each term in the equation to be analytically computable, the objective can be further rewritten to be a combination of several KL-divergence and entropy terms (See the detailed step-by-step process in Appendix B in [Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585)):

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%209.png)

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%2010.png)

## Parameterization of $L_t$ for Training Loss

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%2011.png)

### **Simplification**

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%2012.png)

### **Connection with noise-conditioned score networks (NCSN)**

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%2013.png)

## **Parameterization of $\beta_t$**

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%2014.png)

## **Parameterization of reverse process variance $\varSigma_\theta$**

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%2015.png)

[**SimpleDDPM**](https://colab.research.google.com/drive/1Elj0l93xBJ9xw4oy2fXLUn5C0T0wa5rT#scrollTo=i7AZkYjKgQTm)

[**DDPMCode**](https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main)

# **Speed up Diffusion Model Sampling**

It is very slow to generate a sample from DDPM by following the Markov chain of the reverse diffusion process, as $T$ can be up to one or a few thousand steps. One data point from [Song et al. 2020](https://arxiv.org/abs/2010.02502): “For example, it takes around 20 hours to sample 50k images of size 32 × 32 from a DDPM, but less than a minute to do so from a GAN on an Nvidia 2080 Ti GPU.”

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%2016.png)

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%2017.png)

*Latent diffusion model* (**LDM**; [Rombach & Blattmann, et al. 2022](https://arxiv.org/abs/2112.10752)) runs the diffusion process in the latent space instead of pixel space, making training cost lower and inference speed faster. It is motivated by the observation that most bits of an image contribute to perceptual details and the semantic and conceptual composition still remains after aggressive compression. LDM loosely decomposes the perceptual compression and semantic compression with generative modeling learning by first trimming off pixel-level redundancy with autoencoder and then manipulate/generate semantic concepts with diffusion process on learned latent.

![Fig. 8. The plot for tradeoff between compression rate and distortion, illustrating two-stage compressions - perceptural and semantic comparession. (Image source: [Rombach & Blattmann, et al. 2022](https://arxiv.org/abs/2112.10752))](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%2018.png)

Fig. 8. The plot for tradeoff between compression rate and distortion, illustrating two-stage compressions - perceptural and semantic comparession. (Image source: [Rombach & Blattmann, et al. 2022](https://arxiv.org/abs/2112.10752))

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%2019.png)

![Fig. 9. The architecture of latent diffusion model. (Image source: [Rombach & Blattmann, et al. 2022](https://arxiv.org/abs/2112.1075))](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%2020.png)

Fig. 9. The architecture of latent diffusion model. (Image source: [Rombach & Blattmann, et al. 2022](https://arxiv.org/abs/2112.1075))

## **Conditioned Generation**

While training **generative models on images with conditioning information s**uch as ImageNet dataset, it is common to generate samples **conditioned on class labels or a piece of descriptive text**.

## **Classifier Guided Diffusion**

[](https://arxiv.org/pdf/2105.05233.pdf)

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%2021.png)

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%2022.png)

## **Classifier-Free Guidance**

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%2023.png)

# **Scale up Generation Resolution and Quality**

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%2024.png)

## unCLIp: Dalle

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%2025.png)

Fig. 12. The architecture of unCLIP. (Image source: [Ramesh et al. 2022](https://arxiv.org/abs/2204.06125)])

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%2026.png)

## Imagen

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%2027.png)

![Untitled](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Untitled%2028.png)

### Stable diffusion models

[Stable diffusion models](images/What%20are%20Diffusion%20Models%202cbceaf971814033a4183050ff6ceb35/Stable%20diffusion%20models%20719dcc3061ae4466a2ae5fea4f0cbea8.md)

# **Quick Summary**

- **Pros**: Tractability and flexibility are two conflicting objectives in generative modeling. Tractable models can be analytically evaluated and cheaply fit data (e.g. via a Gaussian or Laplace), but they cannot easily describe the structure in rich datasets. Flexible models can fit arbitrary structures in data, but evaluating, training, or sampling from these models is usually expensive. Diffusion models are both analytically tractable and flexible
- **Cons**: Diffusion models rely on a long Markov chain of diffusion steps to generate samples, so it can be quite expensive in terms of time and compute. New methods have been proposed to make the process much faster, but the sampling is still slower than GAN.
