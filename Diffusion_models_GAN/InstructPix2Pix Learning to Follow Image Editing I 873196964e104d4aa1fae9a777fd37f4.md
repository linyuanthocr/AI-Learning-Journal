# InstructPix2Pix: Learning to Follow Image Editing Instructions

[](https://arxiv.org/pdf/2211.09800.pdf)

## Image Editing from human instructions

We propose a method for editing images from human instructions: given an input image and a written instruction that tells the model what to do, our model follows these instructions to edit the image. To obtain training data for this problem, we combine the knowledge of two large pre- trained models—**a language model (GPT-3) and a text-to- image model (Stable Diffusion)**—to **generate a large dataset of** **image editing examples**. Our **conditional diffusion model**, InstructPix2Pix, is trained on our generated data, and generalizes to real images and user-written instructions at inference time. Since it performs edits in the **forward pass** and does not require per-example fine-tuning or inversion, our model edits images quickly, in a matter of **seconds**. We show compelling editing results for a diverse collection of input images and written instructions.

## **Contribution**

1. **Image editing dataset (Multimodal training dataset)** generation:l GPT 3+ stabble diffusion.
2. Conditional diffusion model based on the **image and instruction only** (does not require any additional example images, full descriptions of the input/output images, or per-example fine- tuning)
3. our model achieves **zero-shot generalization** to both arbitrary real images and natural human-written instructions.

## Method

(1) first, we generate a paired training dataset of text editing instructions and images before/after the edit (Sec. 3.1, Fig. 2a-c), then (2) we train an image editing diffusion model on this generated dataset (Sec. 3.2, Fig 2d). Our model is able to generalize to editing real images using arbitrary human-written instructions. See Fig. 2 for an overview of our method

![Untitled](InstructPix2Pix%20Learning%20to%20Follow%20Image%20Editing%20I%20873196964e104d4aa1fae9a777fd37f4/Untitled.png)

Directional similarities in Clip space

![IMG_3333.png](InstructPix2Pix%20Learning%20to%20Follow%20Image%20Editing%20I%20873196964e104d4aa1fae9a777fd37f4/IMG_3333.png)