# SimCLR:

A Simple Framework for Contrastive Learning of Visual Representations

https://arxiv.org/abs/2002.05709

![image.png](images/SimCLR%201ba71bdab3cf8046b334f4fafe0ca8af/image.png)

1. Augmentation T: sequentially apply three simple augmentations: random cropping followed by resize back to the original size, ran- dom color distortions, and random Gaussian blur.
2. A neural network base encoder f(·): ResNet (output after average pooling layer)
3. A small neural network projection head g(·):  MLP with one hidden layer to obtain
zi = g(hi) = W(2)σ(W(1)hi) where σ is a ReLU non-linearity.

![image.png](images/SimCLR%201ba71bdab3cf8046b334f4fafe0ca8af/image%201.png)

![image.png](images/SimCLR%201ba71bdab3cf8046b334f4fafe0ca8af/image%202.png)

Train:

1. [LARS](https://www.notion.so/LARS-Layer-wise-Adaptive-Rate-Scaling-1ba71bdab3cf806b9c70fef22a142622?pvs=21) optimizer for all batch size (N from 256→8192)
2. Global BN (BN & shuffle on all devices, replacing BN with Layer Norm)
