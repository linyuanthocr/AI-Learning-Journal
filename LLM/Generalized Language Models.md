# Generalized Language Models

[https://lilianweng.github.io/posts/2019-01-31-lm/](https://lilianweng.github.io/posts/2019-01-31-lm/)

# Bert

[https://arxiv.org/pdf/1810.04805.pdf](https://arxiv.org/pdf/1810.04805.pdf)

We have seen amazing progress in NLP in 2018. Large-scale pre-trained language modes like OpenAI GPT and BERT have achieved great performance on a variety of language tasks using generic model architectures. The idea is similar to how ImageNet classification pre-training helps many vision tasks (*). Even better than vision classification pre-training, this simple and powerful approach in NLP does not require labeled data for pre-training, allowing us to experiment with increased training scale, up to our very limit.

# ELMo

**ELMo**, short for **Embeddings from Language Model** ([Peters, et al, 2018](https://arxiv.org/abs/1802.05365)) learns contextualized word representation by pre-training a language model in an *unsupervised* way.

[https://arxiv.org/pdf/1802.05365.pdf](https://arxiv.org/pdf/1802.05365.pdf)

![Untitled](images/Generalized%20Language%20Models%200d446cc7fa09493c879d35570bb17c21/Untitled.png)

![Untitled](images/Generalized%20Language%20Models%200d446cc7fa09493c879d35570bb17c21/Untitled%201.png)

[Bidirectional LSTM review](https://www.notion.so/Bidirectional-LSTM-f373fb0a8a9b4b329ec2588b0e1504c2?pvs=21)

![Untitled](images/Generalized%20Language%20Models%200d446cc7fa09493c879d35570bb17c21/Untitled%202.png)

# GPT

[https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

Following the similar idea of ELMo, OpenAI **GPT**, short for **Generative Pre-training Transformer** ([Radford et al., 2018](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)), expands the unsupervised language model to a much larger scale by training on a giant collection of free text corpora. Despite of the similarity, GPT has two major differences from ELMo.

1. The model architectures are different: ELMo uses a shallow concatenation of independently trained left-to-right and right-to-left multi-layer LSTMs, while GPT is a multi-layer transformer decoder.
2. The use of contextualized embeddings in downstream tasks are different: ELMo feeds embeddings into models customized for specific tasks as additional features, while GPT fine-tunes the same base model for all end tasks.

# Transformer Decoder as Language Model

Compared to the [original transformer](https://arxiv.org/abs/1706.03762) architecture, the [transformer decoder](https://arxiv.org/abs/1801.10198) model discards the encoder part, so there is only one single input sentence rather than two separate source and target sequences.

This model applies multiple transformer blocks over the embeddings of input sequences. Each block contains **a masked *multi-headed self-attention* layer and a *pointwise feed-forward* layer**. The final output produces a distribution over target tokens after softmax normalization.

![Untitled](images/Generalized%20Language%20Models%200d446cc7fa09493c879d35570bb17c21/Untitled%203.png)

![Untitled](images/Generalized%20Language%20Models%200d446cc7fa09493c879d35570bb17c21/Untitled%204.png)

# Byte Pair Encoding

**Byte Pair Encoding** ([**BPE**](https://arxiv.org/abs/1508.07909)) is used to encode the input sequences. BPE was originally proposed as a **data compression algorithm** in 1990s and then was adopted to solve the **open-vocabulary issue** in machine translation, as we can easily run into rare and unknown words when translating into a new language. Motivated by the intuition that rare and unknown words can often be decomposed into multiple subwords, BPE finds the best word segmentation by iteratively and greedily merging frequent pairs of characters.

# **Supervised Fine-Tuning**

The most substantial upgrade that OpenAI GPT proposed is to **get rid of the task-specific model and use the pre-trained language model directly**!

![Untitled](images/Generalized%20Language%20Models%200d446cc7fa09493c879d35570bb17c21/Untitled%205.png)

![Untitled](images/Generalized%20Language%20Models%200d446cc7fa09493c879d35570bb17c21/Untitled%206.png)

**Summary**: It is super neat and encouraging to see that such a general framework is capable to beat SOTA on most language tasks at that time (June 2018). At the first stage, generative pre-training of a language model can absorb as much free text as possible. Then at the second stage, the model is fine-tuned on specific tasks with a small labeled dataset and a minimal set of new parameters to learn.

One limitation of GPT is its uni-directional nature — the model is only trained to predict the future left-to-right context.

# GPT-2

[https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

The [OpenAI](https://blog.openai.com/better-language-models/) [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) language model is a direct successor to [GPT](https://lilianweng.github.io/posts/2019-01-31-lm/#gpt). GPT-2 has 1.5B parameters, 10x more than the original GPT, and it achieves SOTA results on 7 out of 8 tested language modeling datasets in a ***zero-shot transfer setting* without any task-specific fine-tuning**. The pre-training dataset contains 8 million Web pages collected by crawling qualified outbound links from [Reddit](https://www.reddit.com/). Large improvements by OpenAI GPT-2 are s**pecially noticeable on small datasets and datasets used for measuring *long-term dependency***.

# **Zero-Shot Transfer**

The pre-training task for GPT-2 is solely language modeling. All the downstream language tasks are **framed as predicting conditional probabilities** and there is **no task-specific fine-tuning**.

- Text generation is straightforward using LM.
- Machine translation task, for example, English to Chinese, is induced by conditioning LM on pairs of “English sentence = Chinese sentence” and “the target English sentence =” at the end.
    - For example, the conditional probability to predict might look like: `P(? | I like green apples. = 我喜欢绿苹果。 A cat meows at him. = 一只猫对他喵。It is raining cats and dogs. =")`
- QA task is formatted similar to translation with pairs of questions and answers in the context.
- Summarization task is induced by adding `TL;DR:` after the articles in the context.

# BPE on Byte Sequences

Same as the original GPT, GPT-2 uses [BPE](https://lilianweng.github.io/posts/2019-01-31-lm/#byte-pair-encoding) but on [UTF-8](https://en.wikipedia.org/wiki/UTF-8) byte sequences. **Each byte can represent 256 different values in 8 bits, while UTF-8 can use up to 4 bytes for one character, supporting up to 2^31 characters in total.** Therefore, with byte sequence representation we only need a vocabulary of size 256 and do not need to worry about pre-processing, tokenization, etc. Despite of the benefit, current **byte-level LMs** still have non-negligible performance gap with the SOTA **word-level LMs.**

BPE merges frequently co-occurred byte pairs in a greedy manner. To prevent it from generating multiple versions of common words (i.e. `dog.`, `dog!` and `dog?` for the word `dog`), GPT-2 **prevents BPE from merging characters across categories** (thus `dog` would not be merged with punctuations like `.`, `!` and `?`). This tricks help increase the quality of the final byte segmentation.

Using the byte sequence representation, GPT-2 is able to assign a probability to any Unicode string, regardless of any pre-processing steps.

# Model Modifications

Compared to GPT, other than having many more transformer layers and parameters, GPT-2 incorporates only a few architecture modifications:

- [Layer normalization](https://arxiv.org/abs/1607.06450) was moved to the input of each sub-block, similar to a residual unit of type [“building block”](https://arxiv.org/abs/1603.05027) (differently from the original type [“bottleneck”](https://arxiv.org/abs/1512.03385), it has batch normalization applied before weight layers).
- An additional layer normalization was added after the final self-attention block.
- A modified initialization was constructed as a function of the model depth.
- The weights of residual layers were initially scaled by a factor of $1/\sqrt{\mathstrut N}$ where N is the number of residual layers.
- Use larger vocabulary size and context size.

# T5

[https://arxiv.org/pdf/1910.10683.pdf](https://arxiv.org/pdf/1910.10683.pdf)

The language model **T5** is short for **“Text-to-Text Transfer Transformer”** ([Raffel et al., 2020](https://arxiv.org/abs/1910.10683)). The encoder-decoder implementation follows the [original Transformer](https://arxiv.org/abs/1706.03762) architecture: tokens → embedding → encoder → decoder → output. T5 adopts the framework “Natural Language Decathlon” ([McCann et al., 2018](https://arxiv.org/abs/1806.08730)), where many common NLP tasks are translated into question-answering over a context. Instead of an explicit QA format, T5 uses short task prefixes to distinguish task intentions and separately fine-tunes the model on every individual task. The text-to-text framework enables easier transfer learning evaluation with the same model on a diverse set of tasks.

![Untitled](images/Generalized%20Language%20Models%200d446cc7fa09493c879d35570bb17c21/Untitled%207.png)

The model is trained on Web corpus extracted from Apr 2019 with various filters applied. The model is fine-tuned for each downstream task separately via “adapter layers” (add an extra layer for training) or “gradual unfreezing” (see [ULMFiT](https://lilianweng.github.io/posts/2019-01-31-lm/#ulmfit)). Both fine-tuning approaches only update partial parameters while keeping the majority of the model parameters unchanged. T5-11B achieved SOTA results on many NLP tasks.

As the authors mentioned in the paper “…our goal is not to propose new methods but instead to provide a comprehensive perspective on where the field stands”, the T5 long paper described a lot of training setup and evaluation processes in detail, a good read for people who are interested in training a LM from scratch.

# GPT-3

[https://arxiv.org/pdf/2005.14165.pdf](https://arxiv.org/pdf/2005.14165.pdf)

**GPT-3** ([Brown et al., 2020](https://arxiv.org/abs/2005.14165)) has the same architecture as [GPT-2](https://lilianweng.github.io/posts/2019-01-31-lm/#gpt-2) but contains 175B parameters, 10x larger than GPT-2 (1.5B). In addition, GPT-3 uses alternating dense and locally banded sparse attention patterns, same as in [sparse transformer](https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/#sparse-attention-matrix-factorization-sparse-transformers). In order to fit such a huge model across multiple GPUs, GPT-3 is trained with partitions along both width and depth dimension. The training data is a filtered version of Common Crawl mixed with a few other high-quality curated datasets. To avoid the contamination that downstream tasks might appear in the training data, the authors attempted to remove all the overlaps with all the studied benchmark dataset from the training dataset. Unfortunately the filtering process is not perfect due to a bug.

![Untitled](images/Generalized%20Language%20Models%200d446cc7fa09493c879d35570bb17c21/Untitled%208.png)

For all the downstream evaluation, GPT-3 is tested in the few-shot setting without any gradient-based fine-tuning. ***Here the few-shot examples are provided as part of the prompt***. GPT-3 achieves strong performance on many NLP datasets, comparable with fine-tuned BERT models.

![Untitled](images/Generalized%20Language%20Models%200d446cc7fa09493c879d35570bb17c21/Untitled%209.png)

# Instruct GPT (GPT 3.5)

RLHF: reinforcement learning human feedback

![Untitled](images/Generalized%20Language%20Models%200d446cc7fa09493c879d35570bb17c21/Untitled%2010.png)

Technical keypoints:

1. how to collect demonstration data
2. how to collect comparison data
3. how to train a reward model (RM)
4. how to use RM for reinforcement learning

![Untitled](images/Generalized%20Language%20Models%200d446cc7fa09493c879d35570bb17c21/Untitled%2011.png)

![Untitled](images/Generalized%20Language%20Models%200d446cc7fa09493c879d35570bb17c21/Untitled%2012.png)

# 3 Models in GPT 3.5

### Supervised fine-tuning (SFT).

We fine-tune GPT-3 on our labeler demonstrations using supervised learning. We trained for 16 epochs, using a cosine learning rate decay, and residual dropout of 0.2. We do our final SFT model selection based on the RM score on the validation set. Similarly to Wu et al. (2021), we find that our **SFT models overfit on validation loss after 1 epoch**; however, we find that training for more epochs helps both the RM score and human preference ratings, despite this overfitting.

***SFT is used to initialize the RM.***

### Reward modeling (RM).

**Starting from the SFT model with the final unembedding layer removed, we trained a model to take in a prompt and response, and output a scalar reward.** In this paper we only use **6B RMs**, as this saves a lot of compute, and we found that 175B RM training could be unstable and thus was less suitable to be used as the value function during RL (see Appendix C for more details).

![Untitled](images/Generalized%20Language%20Models%200d446cc7fa09493c879d35570bb17c21/Untitled%2013.png)

K = 9, pairwise comparison; (Reason: 1. a little more labeling time, but with huge pair comparison: 2x labeling time, 9x examples. 2. loss calculate efficient: 9 forward & backward calculates for 36 items. 3. less overfitting)

### Reinforcement learning (RL)

PPO(强化学习里面的一种算法)

![Untitled](images/Generalized%20Language%20Models%200d446cc7fa09493c879d35570bb17c21/Untitled%2014.png)

Here using GPT-3 SFT to initialize the RL model, $\pi^{RL}_{\phi} = \pi^{SFT}$.  

$\pi^{RL}_\phi$ is called RL policy (also a probalbilisic distribution).  

When training the RL model, each iterative update of $\phi$ results in the RL model producing a different y for the same prompt x, which means that the environment has changed. $r_\theta$ is the RM model. Equation (2) describes the reinforcement learning process. The pink part is KL reward. The first line of Equation (2) is PPO function. The green part is constrained the generation towards to original data distribution, it’s more like GPT3 loss function.

[Instruct GPT](https://www.notion.so/Instruct-GPT-485a15aef4474679bda9a098b870f6e0?pvs=21)
