# Contrastive learning

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled.png)

# InstDisc

[](https://arxiv.org/pdf/1805.01978.pdf)

contribution：

1. instance discrimination 
2. Memory bank

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%201.png)

Our unsupervised approach takes the class-wise supervision to the extreme and learns **a feature
representation that discriminates among individual instances**

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%202.png)

Encoder and memory bank(draw negative samples) update together. Features are updated with momentum.

**Experiment setting**: we set temperature τ = 0.07 and use Sim with **m = 4, 096** to balance
performance and computing cost. The model is trained for 200 epochs using SGD with momentum. The **batch size is 256**. The learning rate is initialized to 0.03, scaled down with coefficient 0.1 every 40 epochs after the first120 epochs. Our code is available at 

[https://github.com/zhirongw/lemniscate.pytorch](https://github.com/zhirongw/lemniscate.pytorch)

# **Representation Learning with Contrastive Predictive Coding (CPC)**

[](https://arxiv.org/pdf/1807.03748.pdf)

**Generative model**

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%203.png)

$g_{ar}$ is a auto regressive model ,such as RNN or LSTM model

**Pretext task**: prediction method

# **Contrastive Multiview Coding (CMC)**

[](https://arxiv.org/pdf/1906.05849.pdf)

Humans view the world through many sensory channels, e.g., the long-wavelength light channel, viewed by the left eye, or the high-frequency vibrations channel, heard by the right ear. Each view is noisy and incomplete, but **important factors, such as physics, geometry, and semantics, tend to be shared between all views** (e.g., a “dog” can be seen, heard, and felt). We investigate the classic hypothesis that a powerful representation is one that models view-invariant factors. We study this hypothesis under the framework of multiview contrastive learning, where we learn a representation that aims to **maximize mutual information between different views of the same scene** but is otherwise compact. Our approach scales to any number of views, and is view agnostic. We analyze key properties of the approach that make it work, finding that the contrastive loss outperforms a popular alternative based on cross-view prediction, and that the more views we learn from, the better the resulting representation captures underlying scene semantics. Our approach achieves state-of-the-art results on image and video unsupervised learning benchmarks. Code is released at:
[http://github.com/HobbitLong/CMC/](http://github.com/HobbitLong/CMC/).

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%204.png)

**Multiple encoder.**

# MOCO

[MOCO](https://www.notion.so/MOCO-03030c7d88624a16b9fe2ad3b89a583a?pvs=21) 

# **A Simple Framework for Contrastive Learning of Visual Representations (**SimCLR)

[](https://arxiv.org/pdf/2002.05709.pdf)

### Method

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%205.png)

1. A neural network base encoder f(·) that extracts representation vectors from augmented data examples. f(.) is ResNet50, hi dim is 2048.
2. g(.) is a MLP projection head, it gives a 10% up for imageNet classification accuracy.

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%206.png)

1. A contrastive loss function defined for a contrastive prediction task.

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%207.png)

**Algorithm**

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%208.png)

### Contributions

- Composition of **multiple data augmentation operations is crucial** in defining the contrastive prediction tasks that yield effective representations. In addition, unsupervised contrastive learning benefits from **stronger data augmentation** than supervised learning.
- Introducing **a learnable nonlinear transformation** between the representation and the contrastive loss substantially improves the quality of the learned representations.
- Representation learning with **contrastive cross entropy loss** benefits from normalized embeddings and an appropriately adjusted temperature parameter.
- Contrastive learning benefits from **larger batch sizes and longer training** compared to its supervised counterpart. Like supervised learning, contrastive learning benefits from deeper and wider networks.

### Augmentation

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%209.png)

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%2010.png)

**Two best augmentation**: crop and color distort/jitter

### Projection head

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%2011.png)

1. non-linear projection is much better than linear projection and do nothing
2. **Dim 128 is enough**

# Improved Baselines with Momentum Contrastive Learning (MOCOv2)

[](https://arxiv.org/pdf/2003.04297.pdf)

Mocov2 = Moco + simCLR ideas

1. strong augmentation
2. nonlinear projection head
3. cos learning rate schedule
4. more epochs

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%2012.png)

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%2013.png)

**MOCO v2 is better at data exploration.**

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%2014.png)

**MOCO v2 is much more compute effective.**

# Big Self-Supervised Models are Strong Semi-Supervised Learners (SimCLR v2)

[](https://arxiv.org/pdf/2006.10029.pdf)

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%2015.png)

Like noisy student.

1. To fully leverage the power of general pretraining, we **explore larger ResNet models**. Unlike
SimCLR [1] and other previous work [27, 20], whose largest model is ResNet-50 (4×), we train models that are deeper but less wide. The largest model we train is a 152-layer ResNet [25] with 3× wider channels and selective kernels (SK) [28], a channel-wise attention mechanism that improves the parameter efficiency of the network. By scaling up the model from ResNet-50 to **ResNet-152 (3×+SK)**, we obtain a **29% relative improvement** in top-1 accuracy when fine-tuned on 1% of labeled examples.
2. We also **increase the capacity of the non-linear network g(·) (a.k.a. projection head), by making it deeper.**2 Furthermore, instead of throwing away g(·) entirely after pretraining as in SimCLR [1], we **fine-tune from a middle layer (detailed later)**. This small change yields a significant improvement for both linear evaluation and fine-tuning with only a few labeled examples. Compared to SimCLR with **2-layer projection head**, by using a 3-layer projection head and fine-tuning from the 1st layer of projection head, it results in as much as **14% relative improvement** in top-1 accuracy when fine-tuned on 1% of labeled examples (see Figure E.1).
3. Motivated by [29], we also incorporate the **memory mechanism** from MoCo [20], which designates a memory network (with a moving average of weights for stabilization) whose output will be buffered as negative examples. Since our training is based on **large mini-batch** which already supplies many contrasting negative examples, this change yields an improvement of ∼1% for linear evaluation as well as when **fine-tuning on 1%** of labeled examples (see Appendix D).

# Unsupervised Learning of Visual Features by Contrasting Cluster Assignments (SWav)

[](https://arxiv.org/pdf/2006.09882.pdf)

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%2016.png)

c is cluster center, K = 3000.

### Result

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%2017.png)

### Multi-crop: Augment views with smaller image

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%2018.png)

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%2019.png)

# Non negative contrastive method

## **Bootstrap your own latent: A new approach to self-supervised Learning (**BYOL)

[](https://arxiv.org/pdf/2006.07733.pdf)

左脚踩右脚我就上天了！

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%2020.png)

LOSS：MSE LOSS

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%2021.png)

[Understanding self-supervised and contrastive learning with "Bootstrap Your Own Latent" (BYOL)](https://imbue.com/research/2020-08-24-understanding-self-supervised-contrastive-learning/)

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%2022.png)

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%2023.png)

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%2024.png)

[BYOL works even without batch statistics](https://arxiv.org/pdf/2010.10241.pdf)

## **Exploring Simple Siamese Representation Learning（**SimSiam）

[Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566)

### Contributions

simple Siamese networks can learn meaningful representations even using
**none of the following: (i) negative sample pairs, (ii) large batches, (iii) momentum encoders**

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%2025.png)

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%2026.png)

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%2027.png)

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%2028.png)

# An Empirical Study of Training Self-Supervised Vision Transformers (MOCO v3)

[](https://arxiv.org/pdf/2104.02057.pdf)

moco v2+ simsiam

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%2029.png)

**fixed patch projection → better result, stable training**

# **Emerging Properties in Self-Supervised Vision Transformers (**DINO)

[](https://arxiv.org/pdf/2104.14294.pdf)

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%2030.png)

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%2031.png)

![Untitled](images/Contrastive%20learning%20a53ee43e4f30412c81a6927e7086c1a9/Untitled%2032.png)
