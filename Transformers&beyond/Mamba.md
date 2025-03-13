# Mamba

[https://arxiv.org/pdf/2312.00752.pdf](https://arxiv.org/pdf/2312.00752.pdf)

### General

"Mamba: Linear-Time Sequence Modeling with Selective State Spaces," focuses on foundation models, which are central to many applications in deep learning. Traditionally, these models rely heavily on the Transformer architecture and its core attention module. However, the paper identifies a key limitation in many subquadratic-time architectures, including linear attention, gated convolution, recurrent models, and structured state space models (SSMs): their inability to perform content-based reasoning effectively, especially in modalities like language.

To address this, the authors introduce selective SSMs, where the SSM parameters are functions of the input. This approach allows the model to selectively propagate or forget information along the sequence length, depending on the current token. Despite the challenge of forgoing efficient convolutions, the paper presents a hardware-aware parallel algorithm in recurrent mode. The resulting architecture, named Mamba, is streamlined, eliminating the need for attention or MLP blocks.

Mamba demonstrates notable advantages: it has a 5× higher throughput than Transformers and scales linearly in sequence length. Its performance improves on real data, even for sequences up to a million tokens long. Notably, the Mamba-3B model, a specific implementation, outperforms Transformers of the same size and matches the performance of Transformers twice its size in both pretraining and downstream evaluation. Mamba achieves state-of-the-art performance across various modalities, including language, audio, and genomics【8†source】.

Mamba Modeling with **Selective State Spaces,** in which transition from time step to time step is indepent of the input, more like LSTM but remain the backbone as S4 (forward computable by one swoop)

1. ***Training more like transformer, you can calculate with one swoop where you can compute all of the forward passes of the whole sequence in one go. (Not like RNN to compute forward one after another)***
2. ***Inference more like LSTM***

### Architecture

![Fully recurrent model](Mamba%205b46d1e2781c4f9780d1950d556a1a2c/Untitled.png)

Fully recurrent model

![Untitled](Mamba%205b46d1e2781c4f9780d1950d556a1a2c/Untitled%201.png)

### Selective SSM

![Untitled](Mamba%205b46d1e2781c4f9780d1950d556a1a2c/Untitled%202.png)

### **Formula**

![Untitled](Mamba%205b46d1e2781c4f9780d1950d556a1a2c/Untitled%203.png)

### Difference with S4

![Untitled](Mamba%205b46d1e2781c4f9780d1950d556a1a2c/Untitled%204.png)

### Overview of Selective Scan: Hardware-Aware State Expansion

![Untitled](Mamba%205b46d1e2781c4f9780d1950d556a1a2c/Untitled%205.png)

**Prefix Sum**