# Visual generation

# Overview

## Human alignments in visual generation

![Untitled](Visual%20generation%20bd9cf610d0e946e58ff02c22d93bcb93/Untitled.png)

![Untitled](Visual%20generation%20bd9cf610d0e946e58ff02c22d93bcb93/Untitled%201.png)

## Text-to-Image Generation StackGAN

![Untitled](Visual%20generation%20bd9cf610d0e946e58ff02c22d93bcb93/Untitled%202.png)

### Generative adversarial networks (GAN)

GANs (Goodfellow et al., 2020; Creswell et al., 2018; Kang et al., 2023) consist of two key components**: a generator and a discriminator**. The generator is tasked with creating synthetic images from random noise inputs, and it is trained to adjust these noise inputs based on input text conditions to generate semantically relevant images. In this adversarial process, the discriminator competes with the generator, attempting to differentiate between the synthetically generated images and real ones, thus guiding the generator to improve its image creation capabilities

### Variational autoencoder (VAE)

Variational Autoencoder (VAE) (Kingma and Welling, 2013; van den Oord et al., 2017; Vahdat and Kautz, 2020) is a probabilistic model that can generate images by employing **paired encoder and decoder network modules.** The encoder network optimizes the encoding of an image into a **latent representation**, while the decoder refines the process of converting the sampled latent representations back into a new image. VAEs are trained by min- imizing the reconstruction error between the original and decoded images, whileregularizing the encoded latent space using the **Kullback-Leibler (KL) divergence**. Vector Quantised-VAE (VQ- VAE) (van den Oord et al., 2017) further improves VAEs by leveraging the **discrete latent space through vector quantization**, enabling improved reconstruction quality and generative capabilities.

### Discrete image token prediction

At the core of this approach lies a combination of a paired image tokenizer and detokenizer, like Vector Quantized Generative Adversarial Networks (**VQ- GAN**) (Esser et al., 2021), which efficiently transform continuous visual signals into **a finite set of discrete tokens**. In this way, the image generation problem is converted to a **discrete token prediction task**. A widely employed strategy for token prediction is to use an **auto-regressive Transformer** (Ramesh et al., 2021b; Yu et al., 2022b) to sequentially generates visual tokens, typically starting from the top left corner and moving row-by-row towards the bottom right, conditioned on the text inputs. Alternatively, studies (Chang et al., 2022, 2023) also explore the parallel decoding to speed up the token prediction process. Finally, the predicted visual tokens are detokenized, culminating in the final image prediction.

### Diffusion model

Diffusion models (Sohl-Dickstein et al., 2015; Song and Ermon, 2020; Ho et al., 2020) employ stochastic differential equations to evolve random noises into images. A diffusion model works by initiating the process with **a completely random image**, and then gradually refining it over multiple iterations in **a denoising process**. Each iteration predicts and subsequently removes an element of noise, leading to a continuous evolution of the image, conditioned on the input texts

![Untitled](Visual%20generation%20bd9cf610d0e946e58ff02c22d93bcb93/Untitled%203.png)

Stable Diffusion (SD), and its academic version latent diffusion (Rombach et al., 2022), contains mainly three modules, i.e., an **image VAE**, **a denoising U-Net**, **and a condition encoder**, as shown in the left, center, and right part of Figure 3.3, respectively. 

![Untitled](Visual%20generation%20bd9cf610d0e946e58ff02c22d93bcb93/Untitled%204.png)

![Untitled](Visual%20generation%20bd9cf610d0e946e58ff02c22d93bcb93/Untitled%205.png)

# **Spatial Controllable Generation**

[Spatial Controllable Generation](Visual%20generation%20bd9cf610d0e946e58ff02c22d93bcb93/Spatial%20Controllable%20Generation%20357b6ceca40c4a15b45416832f2e7dc7.md)