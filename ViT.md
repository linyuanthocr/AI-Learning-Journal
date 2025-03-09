# ViT

[Vision transformer](https://arxiv.org/pdf/2010.11929.pdf)

[https://arxiv.org/pdf/2010.11929.pdf](https://arxiv.org/pdf/2010.11929.pdf)

![Untitled](ViT%20cc5f6a2354cf4316bbd860faf1cac0b7/Untitled.png)

## **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**

**位置编码是学出来的。196个的768维编码（所有图像相同位置共用一个）。（1D可学习特征，idea来自BERT）**

**ViT 只用了Transformer Encoder部分！！！**

**Class token 也是学习获得的！！所有样本共用一个。**

![Untitled](ViT%20cc5f6a2354cf4316bbd860faf1cac0b7/Untitled%201.png)

**图像patch大小16*16，所以一张224*224图像被变成224/16*224/16个patches，即196个patches，每个patch用长度为768的向量表示。**

**class token: x_class是长度为D的向量，Embedding E和位置编码Epos也是全局共享可以学习的。**

![Untitled](ViT%20cc5f6a2354cf4316bbd860faf1cac0b7/Untitled%202.png)

**Head type and class token. (跟标准transformer学的，其实GAP：global average pooling也可以。)**

[ViT source Code](https://app.yinxiang.com/fx/7b57f39c-55ff-4ed8-8f8b-d2278570624a)