# Image-Only Self-Supervised Learning

Now, we shift our focus to image-only self-supervised learning, and divide the discussion into three parts: (i) contrastive learning, (ii) non-contrastive learning, and (iii) masked image modeling.

# Contrastive and Non-contrastive Learning

### **Contrastive learning**

The core idea of contrastive learning (Gutmann and Hyv¨arinen, 2010; Arora et al., 2019) is to promote the positive sample pairs and repulse the negative sample pairs.

It has been shown that the contrastive objective, known as the **InfoNCE loss** (Oord et al., 2018), can be interpreted as max- imizing the lower bound of mutual information between different views of the data

![Untitled](Image-Only%20Self-Supervised%20Learning%206fb00261a5204842b3ce0e4790e0dc5f/Untitled.png)

In a nutshell, all the image-only contrastive learning methods (e.g., SimCLR (Chen et al., 2020a), see Figure 2.7(a), MoCo (He et al., 2020), SimCLR-v2 (Chen et al., 2020b), MoCo-v2 (Chen et al., 2020c)) share the same high-level framework, detailed below.
• Given one image, two separate data augmentations are applied;  **A base encoder** is followed by **a project head**, which is trained to maximize agreement using a **contrastive loss (i.e., they are from the same image or not);**
• The **project head is thrown away** for downstream tasks.

### Non-contrastive Learning

Recent self-supervised learning methods do not depend on negative samples. The use of negatives is replaced by **asymmetric architectures**

For example, as illustrated in Figure 2.7(b), in SimSiam (Chen and He, 2021), two augmented views of a single image are processed by **an identical encoder network**. Subsequently, a **prediction MLP** is applied to one view, while a **stop-gradient operation** is employed on the other. The primary objective of this model is to **maximize the similarity** between the two views. It is noteworthy that SimSiam relies on neither negative pairs nor a momentum encoder.

**DINO** involves feeding two distinct random transformations of an input image into both the **student and teacher networks**. Both networks share the **same architecture but have different parameters**. The output of the teacher network is centered by computing the mean over the batch. Each network outputs a feature vector that is normalized with a temperature softmax applied to the feature dimension. The similarity between these features is quantified using a cross- entropy loss. Additionally, a stop-gradient operator is applied to the teacher network to ensure that gradients propagate exclusively through the student network. Moreover, DINO updates the teacher’s parameters using an exponential **moving average of the student’s parameters**

# Masked Image Modeling (MIM)

## BEiT

![Untitled](Image-Only%20Self-Supervised%20Learning%206fb00261a5204842b3ce0e4790e0dc5f/Untitled%201.png)

### Image tokenizer

tokenize an image into discrete visual tokens

well-known learning methods for image tokenziers include VQ-VAE (van den Oord et al., 2017), VQ-VAE-2 (Razavi et al., 2019), VQ-GAN (Esser et al., 2021), ViT-VQGAN (Yu et al., 2021), etc.

### Mask-then-predict

Ideas: models accept **the corrupted input image** (e.g., via random masking of image patches), and then **predict the target of the masked content** (e.g., discrete visual tokens in BEiT). As discussed in iBOT (Zhou et al., 2021), this training procedure can be understood as *knowledge distillation between the image tokenizer (which serves as the teacher) and the BEiT encoder (which serves as the student)*, while the student only sees partial of the image.

[**MAE**](CLIP%20&%20CLIP%20Variance%20bf0bbb65afc2404082cbd0a0e204aff8/MAE%20f78945188a5a412ab297ed30a91de285.md) 

### Targets

In Peng et al. (2022b), the authors have provided a unified view of MIM: **a teacher model, a normalization layer, a student model, an MIM head, and a proper loss function**. 

1. **Low-level pixels/features as targets.** leverage either original or normalized pixel values as the target for MIM. finer-grained image understanding.
2. **High-level features as targets.** involve the prediction of discrete tokens using learned image tokenizers. the choice of loss functions depends on the nature of the targets: cross-entropy loss is typically used when the targets are discrete tokens, while ℓ1, ℓ2, or cosine similarity losses are common choices for pixel values or continuous-valued features

### **Lack of learning global image representations**

MIM is an effective pre-training method that provides a good parameter initialization for further model finetuning. However, the vanilla MIM pre-trained model does not learn a global image representation. 

### Scaling properties of MIM

Generally, MIM can be considered an effective regularization method that helps initialize a billion-scale vision transformer for downstream tasks.