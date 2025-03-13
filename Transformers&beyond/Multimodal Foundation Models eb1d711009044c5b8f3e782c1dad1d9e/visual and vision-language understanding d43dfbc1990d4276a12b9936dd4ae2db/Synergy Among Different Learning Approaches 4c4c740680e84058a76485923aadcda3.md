# Synergy Among Different Learning Approaches

### Combining CLIP with label supervision

![Untitled](Synergy%20Among%20Different%20Learning%20Approaches%204c4c740680e84058a76485923aadcda3/Untitled.png)

1. **UniCL** (Yang et al., 2022a) proposes a principled way to use image-label and image-text data together in a joint image-text-label space for unified contrastive learning, and Florence (Yuan et al., 2021) is a scaled-up version of UniCL. 
2. **LiT** (Zhai et al., 2022b) uses a pre-trained ViT-g/14 image encoder learned from supervised pre- training on the JFT-3B dataset, and then makes the image encoder open-vocabulary by learning an additional text tower via contrastive pre-training on image-text data. 
3. **MOFI** (Wu et al., 2023d) proposes to learn image representations from 1 billion noisy entity- annotated images, and uses both image classification and contrastive losses for model training

### Combining CLIP with image-only (non-)contrastive learning.

1. **SLIP** (Mu et al., 2021) proposes a conceptually simple idea to combine **SimCLR** (Chen et al., 2020a) and **CLIP** for model training, and shows that SLIP outperforms CLIP on both zero-shot transfer and linear probe settings. **DeCLIP** (Li et al., 2022g) mines self-supervised learning signals on each modality to make CLIP training data-efficient. In terms of image supervision, the **SimSam** framework (Chen and He, 2021) is used.
2. **xCLIP** (Zhou et al., 2023c) makes CLIP non-contrastive via introducing additional sharpness and smoothness regularization terms borrowed from the image-only non-contrastive learning literature. However, the authors show that only non-contrastive pre-training (nCLIP) is not sufficient to achieve strong performance on zero-shot image classification, and it needs to be combined with the original CLIP for enhanced performance.

### Combining CLIP with MIM

It turns out that image features extracted from CLIP are a good target for MIM training, as the **CLIP image features potentially capture the semantics** that are missing in MIM training. Along

1. **Shallow interaction**
    
    ![Untitled](Synergy%20Among%20Different%20Learning%20Approaches%204c4c740680e84058a76485923aadcda3/Untitled%201.png)
    
2. **Deeper integration**

However, instead of using CLIP as targets for MIM training, if one aims to combine CLIP and MIM for joint model training, MIM does not seem to improve a CLIP model at scale

![Untitled](Synergy%20Among%20Different%20Learning%20Approaches%204c4c740680e84058a76485923aadcda3/Untitled%202.png)