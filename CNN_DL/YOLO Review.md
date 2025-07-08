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

### to be continued
