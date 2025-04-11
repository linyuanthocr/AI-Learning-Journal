# AP-GeM (Attention Pooling Generalized Mean pooling)

The **AP-GeM** (Attention Pooling Generalized Mean pooling) is a technique developed to enhance image retrieval performance by improving how feature vectors are aggregated from convolutional feature maps. It was introduced in the tech report:

> “Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking”
> 

> Filip Radenović, Ahmet Iscen, Giorgos Tolias, Yannis Avrithis, Ondřej Chum
> 

> arXiv:1803.11285
> 

---

**🔍 What is AP-GeM?**

**AP-GeM** extends **GeM pooling** by introducing **attention mechanisms** to better weight and aggregate local image descriptors.

![image.png](images/AP-GeM%20(Attention%20Pooling%20Generalized%20Mean%20pooling%201d271bdab3cf80f2ae99c20811ef66d8/image.png)

---

**🧠 Key Components**

1.	**Backbone**: Commonly based on a CNN like ResNet-101 or ResNet-50, pretrained on ImageNet.

2.	**GeM Pooling**: Aggregates local features into a global descriptor with a learnable exponent p.

3.	**Attention Module**: Learns spatial attention weights over the feature map, improving robustness to clutter and focusing on discriminative regions.

4.	**Whitening**: Optionally applies PCA-whitening to the global descriptor for normalization and improved retrieval performance.

![image.png](images/AP-GeM%20(Attention%20Pooling%20Generalized%20Mean%20pooling%201d271bdab3cf80f2ae99c20811ef66d8/01e2a53b-2455-49ff-bcaf-246e4e12b0bc.png)

**attention weight generation**: This module might consist of a few convolutional layers (often 1x1 convolutions) and the final activation (like Sigmoid).

---

**🚀 Why It Matters**

•	**Improves retrieval**: Especially on datasets like Oxford5k, Paris6k, and Holidays, outperforming average/max pooling and even standard GeM.

•	**Robust to clutter**: Learns to ignore less informative regions (e.g., sky, background).

•	**Plug-and-play**: Can be inserted into most CNN-based retrieval pipelines.

---

**📊 Benchmarks**

In the paper, AP-GeM shows significant improvements in mAP on standard benchmarks:

| **Dataset** | **ResNet101 + GeM** | **ResNet101 + AP-GeM** |
| --- | --- | --- |
| Oxford5k | ~85% | **~89%** |
| Paris6k | ~92% | **~94%** |

---

**📎 Citation**

```
@article{radenovic2018revisiting,
  title={Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking},
  author={Radenovic, Filip and Iscen, Ahmet and Tolias, Giorgos and Avrithis, Yannis and Chum, Ondrej},
  journal={arXiv preprint arXiv:1803.11285},
  year={2018}
}
```

---
