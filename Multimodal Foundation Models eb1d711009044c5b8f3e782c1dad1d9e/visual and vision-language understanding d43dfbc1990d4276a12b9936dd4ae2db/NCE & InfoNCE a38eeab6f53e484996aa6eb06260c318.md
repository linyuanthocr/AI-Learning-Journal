# NCE & InfoNCE

[Contrastive Representation Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/#nce)

# NCE

**Noise Contrastive Estimation**, short for **NCE**, is a method for estimating parameters of a statistical model, proposed by [Gutmann & Hyvarinen](http://proceedings.mlr.press/v9/gutmann10a.html) in 2010. The idea is to run logistic regression to tell apart the target data from noise. Read more on how NCE is used for learning word embedding [here](https://lilianweng.github.io/posts/2017-10-15-word-embedding/#noise-contrastive-estimation-nce).

![Untitled](NCE%20&%20InfoNCE%20a38eeab6f53e484996aa6eb06260c318/Untitled.png)

Here I listed the original form of NCE loss which works with only one positive and one noise sample. In many follow-up works, contrastive loss incorporating multiple negative samples is also broadly referred to as NCE.

# **InfoNCE**

The **InfoNCE loss** in CPC ([Contrastive Predictive Coding](https://lilianweng.github.io/posts/2019-11-10-self-supervised/#contrastive-predictive-coding); [van den Oord, et al. 2018](https://arxiv.org/abs/1807.03748)), inspired by [NCE](https://lilianweng.github.io/posts/2021-05-31-contrastive/#NCE), uses categorical cross-entropy loss to identify the positive sample amongst a set of unrelated noise samples.

![Untitled](NCE%20&%20InfoNCE%20a38eeab6f53e484996aa6eb06260c318/Untitled%201.png)