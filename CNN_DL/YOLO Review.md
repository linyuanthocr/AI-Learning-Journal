# YOLO Review

https://arxiv.org/abs/2304.00501

# YOLOv1 (2016)

![image.png](images/YOLO%20Review%2022871bdab3cf80ec8592d2939069e941/image.png)

![image.png](images/YOLO%20Review%2022871bdab3cf80ec8592d2939069e941/image%201.png)

![image.png](images/YOLO%20Review%2022871bdab3cf80ec8592d2939069e941/image%202.png)

 Leaky rectified relu except the last one used a linear activation function.

### Training process

1. First 20 layers with 224*224 imagenet+classification pretrain. 
2. add last 4 layers with random initialization+ fine tune on 448*448 images

Augmentation: random scaling and translations of at most 20% of the input image size, as well as random exposure and saturation with an upper-end factor of 1.5 in the HSV color space.

![image.png](images/YOLO%20Review%2022871bdab3cf80ec8592d2939069e941/image%203.png)

limits：

1. It could only detect at most two objects of the same class in the grid cell, limiting its ability to predict nearby objects.
2. It struggled to predict objects with **aspect ratios** not seen in the training data. 
3. It learned from coarse object features due to the down-sampling layers.

# YOLOv2: Better, Faster, and Stronger (2017)

### Improvements:

**Batch normalization, high res classifier, fully convolutional. Use anchor boxes (fig 7, one grid several anchors, predict coordinates+classes for each anchor). Dimension clusters (5, k-means based training boxes selection).Direct location prediction(fig 8). Fine-grained features (one less max pooling layer. and passthrough features, Table 2) Multiscale training (no fully connected layer, robust to input size, from 320*320 to 608*608 each 10 batches)**

With all these improvements, YOLOv2 achieved an average precision (AP) of **78.6%** on the PASCAL VOC2007 dataset compared to the **63.4%** obtained by YOLOv1.

![image.png](images/YOLO%20Review%2022871bdab3cf80ec8592d2939069e941/image%204.png)

![image.png](images/YOLO%20Review%2022871bdab3cf80ec8592d2939069e941/image%205.png)

![image.png](images/YOLO%20Review%2022871bdab3cf80ec8592d2939069e941/image%206.png)

The object classification head replaces the last four convolutional layers with a single convolutional layer with 1000 filters, followed by a global average pooling layer and a Softmax.

• Improved to 78.6% AP on PASCAL VOC 2007

# YOLOv3 (2018)

1. Bounding box prediction: logistic regression for object score (gt: 1 for the most overlapping anchor box, 0 for else). one anchor per object. if no anchor box is assigned to the object, only classification loss inferred (no localization loss or confidence loss)

1. class prediction: no softmax, binary cross entropy to train independent logistic regression. multilabel classification.
2. new backbone: 53 layers

![image.png](images/YOLO%20Review%2022871bdab3cf80ec8592d2939069e941/image%207.png)

1. Spatial pyramid pooling(SPP)
2. Multi scale prediction (3 scales * 3 anchors)
3. bounding box priors: kmeans for ground truth. yolov2: 5 anchors per cell. yolov3: 3 anchors in 3 different scales

Darknet-53 (accuracy as resnet152 but 2* faster)

Larger achitecture, muli-scale prediction, (finer detail boxes and significately improved small object detection)

• Achieved 33.0% AP on COCO (benchmark transition from VOC)

### YOLOv3 Multi-Scale

![image.png](images/YOLO%20Review%2022871bdab3cf80ec8592d2939069e941/image%208.png)

For the COCO dataset with 80 categories, **each scale** provides an output tensor with a shape of **N×N×[3×(4+1+80)]** where N ×N is the size of the feature map (or grid cell), the 3 indicates the boxes per cell and the 4 + 1 include the four coordinates and the objectness score.

In **YOLOv3's multi-scale version**, the loss is computed across **three different feature map scales** (usually corresponding to 13×13, 26×26, and 52×52 for input size 416×416), and each of them predicts bounding boxes in the shape:

$$
N×N×[3×(5+C)]
$$

Where:

- N×N = spatial resolution of the feature map
- 3 = number of anchors per scale
- 5+C = 4 box coordinates + 1 objectness score + C class probabilities

---

### ✅ YOLOv3 Multi scale Loss Function (Overview)

The loss for YOLOv3 is computed **per scale**, then summed up:

$$
\text{Total Loss} = \sum_{\text{scale} \in \{13,26,52\}} \text{Loss}_{\text{scale}}
$$

Each **per-scale loss** is composed of:

---

### 📘 1. **Localization Loss (bbox regression)**

For predicted boxes that match ground truth (based on IoU or anchor matching):

$$
\text{Loss}_{\text{loc}} = \lambda_{\text{coord}} \sum_{i} \left[(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 + (w_i - \hat{w}_i)^2 + (h_i - \hat{h}_i)^2 \right]
$$

*Often log-space is used for w, h, and squared error for x, y.*

---

### 📘 2. **Objectness Loss (confidence)**

This is a binary cross-entropy (BCE) loss:

$$
\text{Loss}_{\text{obj}} = \sum_{i} \text{BCE}(p_i^{\text{obj}}, \hat{p}_i^{\text{obj}})
$$

- **Positive anchor** (responsible for a ground-truth box): target = 1
- **Negative anchor** (IoU < threshold, not responsible): target = 0

> ✅ This includes the "unconfidence loss" for background — ensuring the model learns to suppress boxes where no object is present.
> 

---

### 📘 3. **Classification Loss**

Also binary cross-entropy (for multi-label with sigmoid):

$$
\text{Loss}_{\text{cls}} = \sum_{i} \sum_{c=1}^{C} \text{BCE}(p_{i,c}, \hat{p}_{i,c})
$$

Only computed for **positive samples** (anchors matched to ground truth).

---

### 🔁 Multi-scale Loss Calculation Process

For each scale (e.g. 13×13):

1. Decode predictions using anchors and apply sigmoid/log transformations.
2. Match each ground truth box to **the best-fitting anchor** among all scales (based on IoU with anchor sizes).
3. For the best-matched anchor:
    - Mark it as **positive**, and compute all three losses (loc, obj, cls).
4. For all unmatched anchors:
    - Compute **objectness loss only** with target = 0 (negative samples).
5. Repeat for all three scales (13×13, 26×26, 52×52).
6. Sum the losses across all scales.

---

### ⚠️ Notes & Best Practices

- **Each ground truth box is assigned to only one anchor across all scales**—the one with the highest IoU.
- This means:
    - **Localization & classification losses are computed only at that scale**.
    - But **all scales participate in negative objectness loss**, ensuring every cell learns.
    - Final loss is a weighted sum, with tunable hyperparameters like：

$$
\lambda_{\text{coord}}, \lambda_{\text{obj}}
$$

# Backbone, Neck and Head

![image.png](images/YOLO%20Review%2022871bdab3cf80ec8592d2939069e941/image%209.png)

# YOLOv4 (2020)

- **CSPDarknet53** backbone with **cross-stage partial** connections
- **PANet neck** with modified path aggregation
- **Mosaic augmentation** and self-adversarial training
- **Genetic algorithm** hyperparameter optimization
- Evaluated on MS COCO dataset test-dev 2017, YOLOv4 achieved an AP of 43.5% and AP50 of 65.7% at more than 50 FPS on an NVIDIA V100
****

![image.png](images/YOLO%20Review%2022871bdab3cf80ec8592d2939069e941/image%2010.png)

![image.png](images/YOLO%20Review%2022871bdab3cf80ec8592d2939069e941/image%2011.png)

YOLOv4 tried to find the optimal balance by experimenting with many changes categorized as **bag-of-freebies** and **bag-of-specials**. Bag-of-freebies are methods that only change the **training strategy** and increase training cost but do not increase the inference time, the most common being data augmentation. On the other hand, bag-of-specials are methods that **slightly increase the inference cost but significantly improve accurac**y. Examples of these methods are those for enlarging the receptive field [56, 59, 60], combining features [61, 57, 62, 63], and post-processing [64, 49, 65, 66] among others.

**Changes:**

1. Bag-of-Specials (BoS) Integration 
2. Bag-of-freebies (BoF) Integration
3. Self adversarial training (SAT)
4. Hyperparameter optimization with Genetic Algorithm (first 10% of the periods)

![image.png](images/YOLO%20Review%2022871bdab3cf80ec8592d2939069e941/image%2012.png)

### 📘 **CIoU (Complete IoU) Loss**

**CIoU (Complete IoU) Loss** is an improvement over IoU, GIoU, and DIoU losses.

It aims to provide **better convergence and accuracy** by considering:

1. **Overlap area (IoU)**
2. **Distance between centers**
3. **Aspect ratio consistency**

### 🧮 CIoU Loss Formula

Given:

- b: predicted box
- $b^{gt}$: ground truth box
- c: diagonal length of the smallest enclosing box

The **CIoU loss** is defined as:

$$
\mathcal{L}_{CIoU} = 1 - IoU + \frac{\rho^2(\mathbf{b}, \mathbf{b}^{gt})}{c^2} + \alpha v
$$

Where:

- $\rho^2(\mathbf{b}, \mathbf{b}^{gt})$ is the **squared Euclidean distance between the center points**
- c is the **diagonal length of the smallest enclosing box**
- v measures the **difference in aspect ratio**
- $\alpha$ is a positive trade-off parameter

---

🧩 Terms Explained

1. **IoU (Intersection over Union)**
    
    Measures the overlap area between predicted and ground truth boxes.
    
2. **Distance penalty (DIoU term)**
    
    $$
    \frac{\rho^2(\mathbf{b}, \mathbf{b}^{gt})}{c^2}
    $$
    
    Penalizes boxes that are far apart.
    
3. **Aspect ratio penalty (CIoU-only term)**
    
    $$
    v = \frac{4}{\pi^2} \left(\arctan \frac{w^{gt}}{h^{gt}} - \arctan \frac{w}{h} \right)^2
    $$
    
    $$
    \alpha = \frac{v}{(1 - IoU) + v}
    $$
    
    Penalizes differences in width-to-height ratios between predicted and GT boxes.
    

CIoU helps **faster convergence and better accuracy**, especially in tightly-localized object detection tasks.

# YOLOv5

- **PyTorch implementation** by Ultralytics
- **AutoAnchor** algorithm for automatic anchor optimization
- **Five scaled versions** (n, s, m, l, x) for different use cases
- **SPPF layer** for accelerated spatial pyramid pooling
- Achieved 50.7% AP (YOLOv5x) on COCO
- Anchor Free
    - **Each location (grid cell) directly predicts distances**:
        - From an **anchor point** (like the center of the cell) to the four sides of the bounding box (top, left, bottom, right).
        - Using a **DFL** (Distribution Focal Loss) over each distance.
    - **No predefined box sizes or aspect ratios**.
        - Only use a **grid of points** (anchor points), not shaped boxes.
    - **Dynamic label assignment**:
        - During training, each point is matched to GT boxes based on its **proximity and prediction confidence**, not by IoU with an anchor box.

![image.png](images/YOLO%20Review%2022871bdab3cf80ec8592d2939069e941/image%2013.png)

```python
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
```

# **YOLOX (2021)**

- **Anchor-free architecture** returning to original YOLO philosophy
- **Decoupled head** separating classification and regression
- **Advanced label assignment** with simOTA
- **Strong augmentations** eliminating need for ImageNet pretraining
- Achieved 51.2% AP on COCO

# **YOLOv7 (2022)**

- **E-ELAN architecture** for efficient layer aggregation
- **Planned re-parameterized convolution** without identity connections
- **Model scaling strategy** for concatenation-based architectures
- **Auxiliary and lead head** training approach
- Achieved 56.8% AP on COCO (YOLOv7-E6)

# **YOLOv8 (2023)**

- **C2f modules** replacing CSP layers
- **Anchor-free with decoupled head** design
- **Multiple task support** (detection, segmentation, pose estimation)
- **CLI and pip package** integration
- Achieved 53.9% AP (YOLOv8x) on COCO

YOLOv8 uses a similar backbone as YOLOv5 with some changes on the CSPLayer, now called the **C2f module**. The C2f module (**cross-stage partial bottleneck with two convolutions**) combines high-level features with contextual information to improve detection accuracy

![image.png](images/YOLO%20Review%2022871bdab3cf80ec8592d2939069e941/image%2014.png)

- Distribution Focal Loss
    
    In **YOLOv8**, the **DFL (Distribution Focal Loss)** is used to improve bounding box regression by **predicting a distribution over discrete bins** for each box coordinate instead of directly regressing the continuous value. This technique helps the model learn more precise localization.
    
    ---
    
    ### 🧠 What is DFL (Distribution Focal Loss)?
    
    Instead of directly regressing box coordinates like `[x, y, w, h]`, YOLOv8 **discretizes** each coordinate into bins (e.g., `reg_max = 16`, meaning 17 bins from 0 to 16). Then, it:
    
    1. **Predicts a discrete probability distribution** over these bins for each coordinate.
    2. **Applies softmax** to get probabilities.
    3. **Takes the expectation (weighted average)** to obtain the final continuous prediction.
    4. Applies **Distribution Focal Loss** to supervise this distribution.
    
    ---
    
    ### 📐 Coordinate Regression Process
    
    - Each bounding box coordinate (like `tx`, `ty`, `tw`, `th`) is predicted as a **discrete distribution** of length `reg_max + 1` (e.g., 17).
    - The model outputs this as part of its regression head:
        
        ```
        reg_output.shape = [H, W, 4 * (reg_max + 1)]  # for 4 sides of the box
        
        ```
        
    - During training:
        - The **true continuous target** is quantized.
        - The **DFL loss** is computed between the predicted distribution and the quantized ground truth using a focal-style weighting.
    
    ---
    
    ### 🔧 YOLOv8 DFL Loss Equation
    
    Given predicted logits `p = [p_0, p_1, ..., p_16]` and the ground truth bin `t`, the loss is:
    
    DFL(p,t)=FocalLoss(p,t)\text{DFL}(p, t) = \text{FocalLoss}(p, t)
    
    Usually implemented as a **cross-entropy loss**, optionally with focal weights to emphasize harder samples (bins closer to the target value).
    
    ---
    
    ### ✅ Advantages
    
    - Improves localization accuracy, especially for small objects or crowded scenes.
    - Makes the regression task more stable and precise.
    
    ---
    
    ```python
    class DFLoss(nn.Module):
        """Criterion class for computing Distribution Focal Loss (DFL)."""
    
        def __init__(self, reg_max: int = 16) -> None:
            """Initialize the DFL module with regularization maximum."""
            super().__init__()
            self.reg_max = reg_max
    
        def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
            target = target.clamp_(0, self.reg_max - 1 - 0.01)
            tl = target.long()  # target left
            tr = tl + 1  # target right
            wl = tr - target  # weight left
            wr = 1 - wl  # weight right
            return (
                F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
                + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
            ).mean(-1, keepdim=True)
    ```
    
    ```python
    class BboxLoss(nn.Module):
        """Criterion class for computing training losses for bounding boxes."""
    
        def __init__(self, reg_max: int = 16):
            """Initialize the BboxLoss module with regularization maximum and DFL settings."""
            super().__init__()
            self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
    
        def forward(
            self,
            pred_dist: torch.Tensor,
            pred_bboxes: torch.Tensor,
            anchor_points: torch.Tensor,
            target_bboxes: torch.Tensor,
            target_scores: torch.Tensor,
            target_scores_sum: torch.Tensor,
            fg_mask: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Compute IoU and DFL losses for bounding boxes."""
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
    
            # DFL loss
            if self.dfl_loss:
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
                loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
                loss_dfl = loss_dfl.sum() / target_scores_sum
            else:
                loss_dfl = torch.tensor(0.0).to(pred_dist.device)
    
            return loss_iou, loss_dfl
    ```
