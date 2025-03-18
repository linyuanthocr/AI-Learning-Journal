# CLIP variances

# CLIP application in Segmentation

## LANGUAGE-DRIVEN SEMANTIC SEGMENTATION

[](https://arxiv.org/pdf/2201.03546.pdf)

![Untitled](images/CLIP%20variances%208ee0f71a61644e2a8dca0ae57e58be3a/Untitled.png)

![Untitled](images/CLIP%20variances%208ee0f71a61644e2a8dca0ae57e58be3a/Untitled%201.png)

**zero shot segmentation.**

## **GroupViT: Semantic Segmentation Emerges from Text Supervision**

[](https://arxiv.org/pdf/2202.11094.pdf)

![Untitled](images/CLIP%20variances%208ee0f71a61644e2a8dca0ae57e58be3a/Untitled%202.png)

![Untitled](images/CLIP%20variances%208ee0f71a61644e2a8dca0ae57e58be3a/Untitled%203.png)

**Structure**

12 transformer layers.

Inputs : 

1. Image size 224*224, patch 16*16, + ViTSmall, dim=384.  ⇒ 196*384
2. text tokens:64*384, (as cls token, 64 centers) ⇒ 64*384

concatenate ⇒ 260*384

⇒ 6 transformer layers ⇒ grouping block ⇒ 64*384

Add extra 8 tokens ⇒ (64+8)*384

⇒ 3 transformer layers ⇒ grouping block ⇒ 8*384

⇒ 3 transformer layers+MLP ⇒ 1*384

**Clip based contrastive loss.**

![Untitled](images/CLIP%20variances%208ee0f71a61644e2a8dca0ae57e58be3a/Untitled%204.png)

This method has a limitation: it only has 8 group embeddings, so the result can only hold at most 8 classes.

# CLIP variance in object detection

## **Open-vocabulary Object Detection via Vision and Language Knowledge Distillation**

[](https://arxiv.org/pdf/2104.13921.pdf)

![Untitled](images/CLIP%20variances%208ee0f71a61644e2a8dca0ae57e58be3a/Untitled%205.png)

![Untitled](images/CLIP%20variances%208ee0f71a61644e2a8dca0ae57e58be3a/Untitled%206.png)

![Untitled](images/CLIP%20variances%208ee0f71a61644e2a8dca0ae57e58be3a/Untitled%207.png)

![Untitled](images/CLIP%20variances%208ee0f71a61644e2a8dca0ae57e58be3a/Untitled%208.png)

![Untitled](images/CLIP%20variances%208ee0f71a61644e2a8dca0ae57e58be3a/Untitled%209.png)

(c) ViLD-image (CLIP feature as teacher, add open vocabulary)

![Untitled](images/CLIP%20variances%208ee0f71a61644e2a8dca0ae57e58be3a/Untitled%2010.png)

## Grounded Language-Image Pre-training

![Untitled](images/CLIP%20variances%208ee0f71a61644e2a8dca0ae57e58be3a/Untitled%2011.png)

## **CLIP4Clip: An Empirical Study of CLIP for End to End Video Clip Retrieval**

![Untitled](images/CLIP%20variances%208ee0f71a61644e2a8dca0ae57e58be3a/Untitled%2012.png)
