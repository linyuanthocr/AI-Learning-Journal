# DPT：Vision Transformers for Dense Prediction

[https://github.com/isl-org/DPT](https://github.com/isl-org/DPT)

https://huggingface.co/docs/transformers/model_doc/dpt

**Dense prediction transformers**， an architecture that leverages vision transformers in place of **convolutional networks as a backbone** for dense prediction tasks. We **assemble tokens** from **various stages** of the vision transformer into image-like representations at various resolutions and progressively combine them into **full-resolution predictions** using a **convolutional decoder**. 

![image.png](images/DPT%20AVision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image.png)

## Transformer encoder

DPT三种结构：

1. 原始Vit-base, 原始vit输入+12层 transformer layers， D=768
2. 原始Vit-large，原始vit输入+24层 transformer layers， D=1024
3. hybrid resnet50+12层transformer layers

pitchsize 16*16

## Convolutional decoder

three stage

![image.png](images/DPT%20Vision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%201.png)

### **Read**

![image.png](images/DPT%20Vision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%202.png)

3 operations：

![image.png](images/DPT%20Vision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%203.png)

### **Concatenate**

place each token acoording to the position of the initial patch in the image

![image.png](images/DPT%20Vision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%204.png)

### Resample

![image.png](images/DPT%20Vision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%205.png)

we **reassemble features at four different stages** and four differ- ent resolutions. We

combine the extracted feature maps from consecutive stages **using RefineNet-based feature fusion block,** progressively upsample the representation by a factor of two in each fusion stage.

![image.png](images/DPT%20Vision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%206.png)

### Handling varying image size

DPT can **handle varying image sizes**. As long as the image size is divisible by p, the embedding procedure can be applied and will produce a varying number of image tokens Np.

the **position embedding** has a dependency on the image size as it encodes the locations of the patches in the input image. We follow the approach proposed in [11] and **linearly interpolate the position embeddings** to the appropriate size. Note that this can be done on the fly for every image. 

```markdown
# 把 pos_emb 从 (1, N, D) → (1, H, W, D)
pos_emb_2d = pos_emb.reshape(1, H, W, D)
pos_emb_upsampled = interpolate(pos_emb_2d, size=(H_new, W_new), mode='bilinear')
```

### Depth estimation

representations of depth are **unified into a common representation** and that common ambiguities (such as **scale ambiguity**) are appropriately handled in the **training loss** [30].

![image.png](images/DPT%20Vision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%207.png)

a monocular depth prediction network using a **scale- and shift-invariant trimmed loss** that operates on an **inverse depth representation**, together with the **gradient-matching loss** proposed in [22].

1. **scale- and shift-invariant trimmed loss【30】**

![image.png](images/DPT%20Vision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%208.png)

![image.png](images/DPT%20Vision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%209.png)

![image.png](images/DPT%20Vision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%2010.png)

 **b.  gradient-matching loss**

论文【30】的Lreg定义 （逆深度图）

![image.png](images/DPT%20Vision%20Transformers%20for%20Dense%20Prediction%201ba71bdab3cf80f88a58f9790775a2fc/image%2011.png)

- loss code
    
    ```python
    import torch
    import torch.nn as nn
    
    def compute_scale_and_shift(prediction, target, mask):
        # system matrix: A = [[a_00, a_01], [a_10, a_11]]
        a_00 = torch.sum(mask * prediction * prediction, (1, 2))
        a_01 = torch.sum(mask * prediction, (1, 2))
        a_11 = torch.sum(mask, (1, 2))
    
        # right hand side: b = [b_0, b_1]
        b_0 = torch.sum(mask * prediction * target, (1, 2))
        b_1 = torch.sum(mask * target, (1, 2))
    
        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = torch.zeros_like(b_0)
        x_1 = torch.zeros_like(b_1)
    
        det = a_00 * a_11 - a_01 * a_01
        valid = det.nonzero()
    
        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]
    
        return x_0, x_1
    
    def reduction_batch_based(image_loss, M):
        # average of all valid pixels of the batch
    
        # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
        divisor = torch.sum(M)
    
        if divisor == 0:
            return 0
        else:
            return torch.sum(image_loss) / divisor
    
    def reduction_image_based(image_loss, M):
        # mean of average of valid pixels of an image
    
        # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
        valid = M.nonzero()
    
        image_loss[valid] = image_loss[valid] / M[valid]
    
        return torch.mean(image_loss)
    
    def mse_loss(prediction, target, mask, reduction=reduction_batch_based):
    
        M = torch.sum(mask, (1, 2))
        res = prediction - target
        image_loss = torch.sum(mask * res * res, (1, 2))
    
        return reduction(image_loss, 2 * M)
    
    def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):
    
        M = torch.sum(mask, (1, 2))
    
        diff = prediction - target
        diff = torch.mul(mask, diff)
    
        grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
        mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
        grad_x = torch.mul(mask_x, grad_x)
    
        grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
        mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
        grad_y = torch.mul(mask_y, grad_y)
    
        image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))
    
        return reduction(image_loss, M)
    
    class MSELoss(nn.Module):
        def __init__(self, reduction='batch-based'):
            super().__init__()
    
            if reduction == 'batch-based':
                self.__reduction = reduction_batch_based
            else:
                self.__reduction = reduction_image_based
    
        def forward(self, prediction, target, mask):
            return mse_loss(prediction, target, mask, reduction=self.__reduction)
    
    class GradientLoss(nn.Module):
        def __init__(self, scales=4, reduction='batch-based'):
            super().__init__()
    
            if reduction == 'batch-based':
                self.__reduction = reduction_batch_based
            else:
                self.__reduction = reduction_image_based
    
            self.__scales = scales
    
        def forward(self, prediction, target, mask):
            total = 0
    
            for scale in range(self.__scales):
                step = pow(2, scale)
    
                total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                       mask[:, ::step, ::step], reduction=self.__reduction)
    
            return total
    
    class ScaleAndShiftInvariantLoss(nn.Module):
        def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
            super().__init__()
    
            self.__data_loss = MSELoss(reduction=reduction)
            self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
            self.__alpha = alpha
    
            self.__prediction_ssi = None
    
        def forward(self, prediction, target):
            #preprocessing
            mask = target > 0
    
            #calcul
            scale, shift = compute_scale_and_shift(prediction, target, mask)
            # print(scale, shift)
            self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
    
            total = self.__data_loss(self.__prediction_ssi, target, mask)
            if self.__alpha > 0:
                total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)
    
            return total
    
        def __get_prediction_ssi(self):
            return self.__prediction_ssi
    
        prediction_ssi = property(__get_prediction_ssi)
    ```
    

论文【30】

https://arxiv.org/abs/1907.01341

论文【22】

https://arxiv.org/abs/1804.00607
