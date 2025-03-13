# S4 model: Long Sequences with Structured State Spaces

[https://arxiv.org/pdf/2111.00396.pdf](https://arxiv.org/pdf/2111.00396.pdf)

The paper primarily focuses on addressing the challenge of efficiently modeling long sequences in deep learning. The paper introduces a new approach called the Structured State Space (S4) model, which shows significant improvement in handling long-range dependencies in sequence modeling. Key contributions and findings of this paper include:

1. **Problem Identification**: The paper addresses the difficulty of modeling long sequences with conventional models like RNNs, CNNs, and Transformers, which struggle to scale to very long sequences (e.g., more than 10,000 steps).
2. **Introduction of S4 Model**: The S4 model is proposed based on a new parameterization of the State Space Model (SSM). This model overcomes the computational and memory inefficiencies of prior methods while preserving their theoretical strengths.
3. **Efficiency of S4**: The paper demonstrates that S4 can be computed much more efficiently than previous approaches. It uses a **low-rank correction and diagonalization, reducing the SSM to a computation of a Cauchy kernel.**
4. **Empirical Results**: S4 achieves strong empirical results across a range of benchmarks. Notably, it significantly outperforms other models in the Long Range Arena benchmark, solving tasks like the Path-X of length 16k, which prior models failed at.
5. **Applications and Implications**: This research has substantial implications for sequence modeling, particularly in handling long-range dependencies. It showcases S4's potential as a general sequence model applicable across various domains like images, text, and time-series.
6. **Comparison with Other Models**: S4 shows superior performance compared to efficient Transformer variants, often with faster computation and less memory usage. It also demonstrates robustness across different tasks, such as sequential image classification, language modeling, and time-series forecasting.
7. **Theoretical Contributions**: The paper includes significant theoretical contributions, like the S4 convolution kernel algorithm, which allows efficient computation and training of the model.

The paper concludes that S4 represents a significant advancement in sequence modeling, especially in efficiently handling long-range dependencies. This advancement could potentially influence a wide range of applications in machine learning and deep learning, particularly in natural language processing and time-series analysis.

**Our technique involves conditioning A with a low-rank correction, allowing it to be diagonalized stably and reducing the SSM to the well-studied computation of a Cauchy kernel.**

S4 properties:

1. linear transaction between $h_{t-1}$ to  $h_t$ 
2. no time dependence between ( $h_{t-1}$ to  $h_t$ ) or ( $h_{t-1}$ to  $h_{t-2}$ )