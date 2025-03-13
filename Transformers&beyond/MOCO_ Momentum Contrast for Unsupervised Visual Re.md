# MOCO_ Momentum Contrast for Unsupervised Visual Representation Learning

https://medium.com/@noureldinalaa93/easily-explained-momentum-contrast-for-unsupervised-visual-representation-learning-moco-c6f00a95c4b2

[https://github.com/facebookresearch/moco](https://github.com/facebookresearch/moco)

[arxiv.org](https://arxiv.org/pdf/1911.05722)

The **MoCo paper**, titled "Momentum Contrast for Unsupervised Visual Representation Learning," introduces a novel method for unsupervised visual representation learning. It leverages a **contrastive learning** approach with a **momentum-based encoder** to build a large and consistent dictionary for visual representation. Essentially, it learns to represent images by contrasting them with other images, without needing labeled data.

The paper highlights the importance of a large and consistent dictionary in contrastive learning. Previous methods were limited by either memory capacity or the use of outdated representations. MoCo addresses this by maintaining a **queue** of encoded representations and using a **momentum encoder** to update the keys in the queue. This allows **MoCo to utilize a significantly larger dictionary** than previous methods while ensuring its consistency

As you can see in the diagram, MoCo uses two encoders: a **query encoder** and a **key encoder**. The query encoder processes the current input image, while the key encoder processes the images in the queue. The momentum encoder updates the key encoder slowly, preventing drastic changes and ensuring consistency in the dictionary.

![image.png](images/MOCO_%20Momentum%20Contrast%20for%20Unsupervised%20Visual%20Re%2014571bdab3cf8064948ec7c9575c2757/image.png)

### **1. Key Idea**

MoCo builds a **dynamic dictionary** with a queue and a momentum-based encoder to facilitate contrastive learning. The core idea is to maintain a consistent set of **negative samples** across iterations, even when using small batch sizes, thereby decoupling the number of negatives from the batch size.

![image.png](images/MOCO_%20Momentum%20Contrast%20for%20Unsupervised%20Visual%20Re%2014571bdab3cf8064948ec7c9575c2757/image%201.png)

![image.png](images/MOCO_%20Momentum%20Contrast%20for%20Unsupervised%20Visual%20Re%2014571bdab3cf8064948ec7c9575c2757/image%202.png)

![image.png](images/MOCO_%20Momentum%20Contrast%20for%20Unsupervised%20Visual%20Re%2014571bdab3cf8064948ec7c9575c2757/image%203.png)

![image.png](images/MOCO_%20Momentum%20Contrast%20for%20Unsupervised%20Visual%20Re%2014571bdab3cf8064948ec7c9575c2757/image%204.png)

![image.png](images/MOCO_%20Momentum%20Contrast%20for%20Unsupervised%20Visual%20Re%2014571bdab3cf8064948ec7c9575c2757/image%205.png)

![image.png](images/MOCO_%20Momentum%20Contrast%20for%20Unsupervised%20Visual%20Re%2014571bdab3cf8064948ec7c9575c2757/image%206.png)
