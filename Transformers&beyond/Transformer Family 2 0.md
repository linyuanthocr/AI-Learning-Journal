# Transformer Family 2.0

[https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled.png)

# **Transformer Basics**

The **Transformer** (which will be referred to as “vanilla Transformer” to distinguish it from other enhanced versions; [Vaswani, et al., 2017](https://arxiv.org/abs/1706.03762)) model has an encoder-decoder architecture, as commonly used in many [NMT](https://lilianweng.github.io/posts/2018-06-24-attention/#born-for-translation) models. Later simplified Transformer was shown to achieve great performance in language modeling tasks, like in encoder-only [BERT](https://lilianweng.github.io/posts/2019-01-31-lm/#bert) or decoder-only [GPT](https://lilianweng.github.io/posts/2019-01-31-lm/#openai-gpt).

## **Attention and Self-Attention**

**Attention** is a mechanism in neural network that a model can learn to make predictions by selectively attending to a given set of data. The amount of attention is quantified by learned weights and thus the output is usually formed as a weighted average.

**Self-attention** is a type of attention mechanism where the model makes prediction for one part of a data sample using other parts of the observation about the same sample. Conceptually, it feels quite similar to [non-local means](https://en.wikipedia.org/wiki/Non-local_means). Also note that self-attention is **permutation-invariant**; in other words, it is an operation on sets.

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%201.png)

## Multi-Head Self-Attention

The **multi-head self-attention** module is a key component in Transformer. Rather than only computing the attention once, the multi-head mechanism splits the inputs into smaller chunks and then computes the **scaled dot-product attention** over each subspace in parallel. The independent attention outputs are simply concatenated and linearly transformed into expected dimensions.

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%202.png)

## Encoder-Decoder Architecture

The **encoder** generates an attention-based representation with capability to locate a specific piece of information from a large context. It consists of a stack of 6 identity modules, each containing two submodules, a ***multi-head self-attention*** layer and a ***point-wise* fully connected feed-forward network**. By point-wise, it means that it applies the same linear transformation (with same weights) to each element in the sequence. This can also be viewed as a convolutional layer with filter size 1. Each submodule has a residual connection and layer normalization. All the submodules **output data of the same dimension $d$.**

The function of Transformer **decoder** is to retrieve information from the encoded representation. The architecture is quite similar to the encoder, except that the decoder contains ****two multi-head attention submodules instead of one** in each identical repeating module. The first multi-head attention submodule is masked to prevent positions from attending to the future.

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%203.png)

## **Positional Encoding**

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%204.png)

### **Learned Positional Encoding**

Learned positional encoding assigns each element with a *learned* column vector which encodes its absolute position ([Gehring, et al. 2017](https://arxiv.org/abs/1705.03122)) and furthermroe this encoding can be learned differently per layer ([Al-Rfou et al. 2018](https://arxiv.org/abs/1808.04444)).

### **Relative Position Encoding**

[https://arxiv.org/pdf/1803.02155.pdf](https://arxiv.org/pdf/1803.02155.pdf)

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%205.png)

Original Self attention:

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%206.png)

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%207.png)

Proposed one: 

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%208.png)

### Transformer-XL

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%209.png)

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%2010.png)

[https://arxiv.org/pdf/1901.02860.pdf](https://arxiv.org/pdf/1901.02860.pdf)

We propose a novel neural architecture Transformer-XL that enables learning dependency beyond a fixed length without disrupting temporal coherence. It consists of a segment-level recurrence mechanism and a novel positional encoding scheme. Our method not only enables capturing longer-term dependency, but also resolves the context fragmentation problem. As a result, Transformer-XL learns dependency that is 80% longer than RNNs and 450% longer than vanilla Transformers, achieves better performance on both short and long sequences, and is up to 1,800+ times faster than vanilla Transformers during evaluation.

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%2011.png)

### **Rotary Position Embedding**

[https://arxiv.org/pdf/2104.09864.pdf](https://arxiv.org/pdf/2104.09864.pdf)

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%2012.png)

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%2013.png)

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%2014.png)

[十分钟读懂旋转编码（RoPE）](https://zhuanlan.zhihu.com/p/647109286)

[RoPE](https://www.notion.so/RoPE-6b6011b62fd34c429704ca167fc53493?pvs=21)

# **Longer Context**

The length of an input sequence for transformer models at inference time is upper-bounded by the context length used for training. Naively increasing context length leads to high consumption in both time $(O(L^2d))$ and memory $(O(L^2))$ and may not be supported due to hardware constraints.

This section introduces several improvements in transformer architecture to better support long context at inference; E.g. using additional memory, design for better context extrapolation, or recurrency mechanism.

## **Context Memory**

The vanilla Transformer has a fixed and limited attention span. The model can only attend to other elements in the same segments during each update step and no information can flow across separated fixed-length segments. This *context segmentation* causes several issues:

- The model cannot capture very long term dependencies.
- It is hard to predict the first few tokens in each segment given no or thin context.
- The evaluation is expensive. Whenever the segment is shifted to the right by one, the new segment is re-processed from scratch, although there are a lot of overlapped tokens.

**Transformer-XL** ([Dai et al., 2019](https://arxiv.org/abs/1901.02860); “XL” means “extra long”) modifies the architecture to reuse hidden states between segments with an additional memory. The recurrent connection between segments is introduced into the model by continuously using the hidden states from the previous segments.

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%2015.png)

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%2016.png)

[https://arxiv.org/pdf/1911.05507.pdf](https://arxiv.org/pdf/1911.05507.pdf)

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%2017.png)

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%2018.png)

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%2019.png)

# Efficient Attention

The computation and memory cost of the vanilla Transformer grows quadratically with sequence length and hence it is hard to be applied on very long sequences. Many efficiency improvements for Transformer architecture have something to do with the self-attention module - making it cheaper, smaller or faster to run. See the survey paper on *Efficient Transformers* ([Tay et al. 2020](https://arxiv.org/abs/2009.06732)).

## Sparse Attention Patterns

### Fixed Local Context

A simple alternation to make self-attention less expensive is to restrict the attention span of each token to **local** context only, so that self-attention grows linearly with the sequence length.

The idea was introduced by **Image Transformer** ([Parmer, et al 2018](https://arxiv.org/abs/1802.05751)), which formulates image generation as sequence modeling using an encoder-decoder transformer architecture:

- The encoder generates a contextualized, per-pixel-channel representation of the source image;
- Then the decoder autoregressively generates an output image, one channel per pixel at each time step.

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%2020.png)

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%2021.png)

### **Strided Context**

**Sparse Transformer** ([Child et al., 2019](https://arxiv.org/abs/1904.10509)) introduced *factorized self-attention*, through sparse matrix factorization, making it possible to train dense attention networks with hundreds of layers on sequence length up to 16,384, which would be infeasible on modern hardware otherwise.

[https://arxiv.org/pdf/1904.10509.pdf](https://arxiv.org/pdf/1904.10509.pdf)

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%2022.png)

**Sparse Factorized Attention**

Sparse Transformer proposed two types of **fractorized attention**. It is easier to understand the concepts as illustrated in Fig. 10 with 2D image inputs as examples.

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%2023.png)

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%2024.png)

### **Use Factorized Self-Attention in Transformer**

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%2025.png)

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%2026.png)

About the 3rd way

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%2027.png)

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%2028.png)

Sparse Transformer also proposed a set of changes so as to train the Transformer up to **hundreds of layers**, including **gradient checkpointing, recomputing attention & FF layers during the backward pass, mixed precision training, efficient block-sparse implementation, etc.** Please check the [paper](https://arxiv.org/abs/1904.10509) for more details or my previous post on [techniques for scaling up model training](https://lilianweng.github.io/posts/2021-09-25-train-large/).

Process:

![Untitled](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Untitled%2029.png)

### Efficient block-sparse attention

The sparse attention masks in 3(b) and 3(c) can be efficiently computed by slicing out sub-blocks from the query, key, and value matrices and computing the product in blocks. Attention over a local window can be computed as-is, whereas attention with a stride of k can be computed by transposing the matrix and computing a local window. Fixed attention positions can be aggregated and computed in blocks.
In order to ease experimentation, we implemented a set of GPU kernels which efficiently perform these operations. The softmax operation is fused into a single kernel and also uses registers to eliminate loading the input data more than once, allowing it to run at the same speed as a simple nonlinearity. The upper triangle of the attention matrix is never computed, moreover, **removing the need for the negative bias term of (Vaswani et al., 2017) and halving the number of operations to be performed.**

### Mixed-precision training

We store **network weights in single-precision floating-point, but otherwise compute network activations and gradients in half-precision**, as in (Micikevicius et al., 2017). This accelerates our training due to the usage of Tensor Core operations on the V100 GPU. During the gradient calculation, we use dynamic loss scaling to reduce numerical underflow, and we communicate half-precision gradients when averaging across multiple GPUs. When sampling, we cast the queries and keys to single-precision, as the query-key product can sometimes overflow the max value of half-precision.

[Swin Transformer](images/Transformer%20Family%202%200%20914e4aab3f78490bbf8769d797f92961/Swin%20Transformer%203f145133347d4864bad610512b237b2d.md)
