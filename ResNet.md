# ResNet

![image.png](images/ResNet%201b471bdab3cf809085abd9f0a49fdbca/image.png)

![image.png](images/ResNet%201b471bdab3cf809085abd9f0a49fdbca/image%201.png)

### ResNet (Residual Network): A Professional Overview

### **Introduction**

ResNet (Residual Network) is a deep convolutional neural network (CNN) architecture introduced by **He et al.** in the 2015 paper *"Deep Residual Learning for Image Recognition"* (CVPR 2016). It was designed to overcome the **vanishing gradient problem** that hampers the training of very deep neural networks. ResNet achieved groundbreaking performance in computer vision tasks, winning the **ILSVRC 2015 classification competition**.

---

### **1. Key Architectural Innovations**

### **1.1 Residual Learning**

The central idea behind ResNet is the **residual connection** (or skip connection). Instead of learning a direct mapping H(x)H(x) from input xx to output, ResNet models learn the **residual function**:

F(x)=H(x)−xF(x) = H(x) - x

Rearranging, the original function can be rewritten as:

H(x)=F(x)+xH(x) = F(x) + x

This transformation is implemented using **shortcut connections**, which directly bypass one or more layers, allowing the model to retain gradient flow during backpropagation.

### **1.2 Identity Shortcut Connections**

The skip connections allow the model to propagate gradients more effectively through deep networks, significantly reducing the risk of vanishing gradients. The standard residual block is:

y=F(x,W)+xy = F(x, W) + x

where:

- xx is the input,
- WW represents the weights of convolutional layers,
- F(x,W)F(x, W) is the learned transformation, typically comprising **two or three convolutional layers** followed by batch normalization (BN) and activation functions.

When the identity mapping is used (i.e., **no learnable parameters in the shortcut connection**), the network training becomes more stable.

---

### **2. ResNet Variants**

Several versions of ResNet have been introduced, each differing in depth:

| Model | Architecture |
| --- | --- |
| **ResNet-18** | 18 layers deep |
| **ResNet-34** | 34 layers deep |
| **ResNet-50** | 50 layers deep, uses bottleneck layers |
| **ResNet-101** | 101 layers deep |
| **ResNet-152** | 152 layers deep |

### **2.1 Basic vs. Bottleneck Blocks**

- **Basic Block** (used in ResNet-18 and ResNet-34):
    
    Conv(3x3) → BN → ReLU → Conv(3x3) → BN + Skip Connection\text{Conv(3x3) → BN → ReLU → Conv(3x3) → BN + Skip Connection}
    
- **Bottleneck Block** (used in ResNet-50, ResNet-101, ResNet-152):
    
    Conv(1x1) → BN → ReLU → Conv(3x3) → BN → ReLU → Conv(1x1) → BN + Skip Connection\text{Conv(1x1) → BN → ReLU → Conv(3x3) → BN → ReLU → Conv(1x1) → BN + Skip Connection}
    
    - The **bottleneck** design reduces computation by first reducing the number of channels before applying 3x3 convolutions.

---

### **3. Training Advantages of ResNet**

- **Better Gradient Flow**: Residual connections allow gradients to propagate deeper without degradation.
- **Faster Convergence**: ResNets train faster than plain deep CNNs.
- **Enables Extremely Deep Networks**: Traditional deep networks (e.g., VGG) suffer from degradation when depth increases beyond a certain limit. ResNet solves this issue, allowing models with 100+ layers.

---

### **4. Applications of ResNet**

ResNet has been widely adopted in various deep learning tasks, including:

- **Image Classification** (e.g., ResNet pre-trained on ImageNet)
- **Object Detection** (e.g., Faster R-CNN with ResNet backbone)
- **Semantic Segmentation** (e.g., DeepLabV3+ using ResNet)
- **Medical Image Analysis** (e.g., MRI and CT scan classification)
- **3D Vision & Video Analysis** (e.g., ResNet-3D for action recognition)

---

### **5. Improvements Over ResNet**

Several architectures have built upon ResNet to improve its efficiency:

- **ResNeXt**: Uses grouped convolutions for increased efficiency.
- **EfficientNet**: Uses compound scaling instead of residual blocks.
- **DenseNet**: Uses feature reuse across layers instead of residual connections.
- **RegNet**: Learns the best network width and depth dynamically.

---

### **6. Code Implementation (PyTorch)**

A simple ResNet block implementation in PyTorch:

```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Identity mapping
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Residual connection
        out = self.relu(out)
        return out

```

This forms the foundation of a full ResNet implementation, where multiple such blocks are stacked to build deeper models.

---

### **7. Summary**

- **ResNet introduced skip connections** to address the degradation problem in deep neural networks.
- **Residual blocks enable stable training of deep architectures** (e.g., ResNet-50, ResNet-101, ResNet-152).
- **Variants like ResNeXt, DenseNet, and EfficientNet build upon ResNet’s success**.
- **ResNet remains a powerful backbone for many vision tasks**, including classification, detection, and segmentation.

Would you like further details on a specific aspect, such as optimization techniques, transfer learning, or comparisons with other architectures?

Here are implementations of **ResNet-50** in both **TensorFlow (Keras)** and **PyTorch**.

---

## **1. TensorFlow/Keras Implementation of ResNet-50**

```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model

def residual_block(x, filters, strides=1, identity_downsample=None):
    """Creates a bottleneck residual block with identity shortcut"""
    x_shortcut = x

    # First 1x1 convolution (reduces channels)
    x = Conv2D(filters, kernel_size=1, strides=strides, padding='valid', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second 3x3 convolution (feature extraction)
    x = Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Third 1x1 convolution (expands channels)
    x = Conv2D(4 * filters, kernel_size=1, strides=1, padding='valid', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Identity shortcut connection
    if identity_downsample is not None:
        x_shortcut = identity_downsample(x_shortcut)

    x = Add()([x, x_shortcut])
    x = ReLU()(x)

    return x

def build_resnet50(input_shape=(224, 224, 3), num_classes=1000):
    """Creates the ResNet-50 model"""
    inputs = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # ResNet-50 Bottleneck Architecture
    filters = [64, 128, 256, 512]
    repetitions = [3, 4, 6, 3]  # Number of bottleneck blocks per stage

    for i in range(4):
        for j in range(repetitions[i]):
            strides = 1 if j > 0 else 2  # First block in each stage downsamples
            identity_downsample = None
            if j == 0:  # Downsampling only for first block in each stage
                identity_downsample = tf.keras.Sequential([
                    Conv2D(4 * filters[i], kernel_size=1, strides=strides, use_bias=False),
                    BatchNormalization()
                ])
            x = residual_block(x, filters[i], strides, identity_downsample)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Instantiate and compile the model
resnet50_keras = build_resnet50()
resnet50_keras.summary()

```

---

## **2. PyTorch Implementation of ResNet-50**

```python
import torch
import torch.nn as nn

class BottleneckBlock(nn.Module):
    """ResNet-50 Bottleneck Residual Block"""
    expansion = 4  # Bottleneck expands filters by a factor of 4

    def __init__(self, in_channels, filters, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)

        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

        self.conv3 = nn.Conv2d(filters, filters * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # To match dimensions in case of stride change

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += identity  # Add shortcut connection
        x = self.relu(x)

        return x

class ResNet50(nn.Module):
    """ResNet-50 Model"""
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Define layers
        self.layer1 = self._make_layer(64, 3, stride=1)  # 3 Bottleneck blocks
        self.layer2 = self._make_layer(128, 4, stride=2)  # 4 Bottleneck blocks
        self.layer3 = self._make_layer(256, 6, stride=2)  # 6 Bottleneck blocks
        self.layer4 = self._make_layer(512, 3, stride=2)  # 3 Bottleneck blocks

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BottleneckBlock.expansion, num_classes)

    def _make_layer(self, filters, num_blocks, stride):
        """Create a ResNet Layer (group of Bottleneck Blocks)"""
        downsample = None
        if stride != 1 or self.in_channels != filters * BottleneckBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, filters * BottleneckBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(filters * BottleneckBlock.expansion)
            )

        layers = []
        layers.append(BottleneckBlock(self.in_channels, filters, stride, downsample))
        self.in_channels = filters * BottleneckBlock.expansion
        for _ in range(1, num_blocks):
            layers.append(BottleneckBlock(self.in_channels, filters))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of ResNet-50"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Instantiate the model
resnet50_pytorch = ResNet50(num_classes=1000)
print(resnet50_pytorch)

```

---

## **Comparison of TensorFlow and PyTorch Implementations**

| Feature | TensorFlow (Keras) | PyTorch |
| --- | --- | --- |
| **Framework** | TensorFlow 2.x (Keras API) | PyTorch |
| **Layer Definition** | Functional API (Keras) | nn.Module subclasses |
| **Training** | Uses `model.fit()` | Uses `torch.optim` and custom training loops |
| **Dynamic Computation** | Static graph computation (TF 2.x supports eager execution) | Dynamic computation graphs |

### **Which One to Use?**

- **Use TensorFlow** if you want an easy-to-use high-level API (`model.fit()`, `model.predict()`).
- **Use PyTorch** if you prefer fine-grained control and a more Pythonic way to define models.

Would you like an example of training these models on ImageNet or CIFAR-10? 🚀
