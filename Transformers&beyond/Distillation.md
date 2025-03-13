# Distillation

Distilling the Knowledge in a Neural Network

![IMG_1932.jpeg](images/Distillation%209d43d07c029d4eaebc8aac0ca6f85701/IMG_1932.jpeg)

![IMG_1933.jpeg](images/Distillation%209d43d07c029d4eaebc8aac0ca6f85701/IMG_1933.jpeg)

**结论**

知识蒸馏，可以将一个网络的知识转移到另一个网络，两个网络可以是同构或者异构。做法是先训练一个teacher网络，然后使用这个teacher网络的输出和数据的真实标签去训练student网络。知识蒸馏，可以用来将网络从大网络转化成一个小网络，并保留接近于大网络的性能；也可以将多个网络的学到的知识转移到一个网络中，使得单个网络的性能接近emsemble的结果。

参考文献：

1. Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network."*arXiv preprint arXiv:1503.02531*(2015).
2. Prakhar Ganesh. "Knowledge Distillation : Simplified". [https://towardsdatascience.com/knowledge-distillation-simplified-dd4973dbc764](https://link.zhihu.com/?target=https%3A//towardsdatascience.com/knowledge-distillation-simplified-dd4973dbc764), 2019.

[【经典简读】知识蒸馏(Knowledge Distillation) 经典之作](https://zhuanlan.zhihu.com/p/102038521?utm_psn=1734340205249839104)
