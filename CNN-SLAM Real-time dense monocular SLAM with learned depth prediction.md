# CNN-SLAM: Real-time dense monocular SLAM with learned depth prediction

https://arxiv.org/abs/1704.03489

[https://github.com/iitmcvg/CNN_SLAM](https://github.com/iitmcvg/CNN_SLAM)

![image.png](images/CNN-SLAM%20Real-time%20dense%20monocular%20SLAM%20with%20learn%201b471bdab3cf8032acd7f0ecedd731a4/image.png)

![image.png](images/CNN-SLAM%20Real-time%20dense%20monocular%20SLAM%20with%20learn%201b471bdab3cf8032acd7f0ecedd731a4/image%201.png)

**CNN-SLAM: Real-time Dense Monocular SLAM with Learned Depth Prediction** is a research paper that explores the combination of deep learning-based depth prediction with traditional SLAM (Simultaneous Localization and Mapping) methods. Below is a summary of its key contributions:

### **Key Concepts**

1. **Monocular SLAM**: Traditional SLAM methods using a single camera suffer from scale drift and challenges in depth estimation due to lack of direct depth information.
2. **Depth Prediction via CNNs**: The paper leverages convolutional neural networks (CNNs) trained on large datasets to predict depth from a single image.
3. **Fusion of Deep Learning & SLAM**: It integrates CNN-based depth estimation into a SLAM pipeline to improve depth accuracy and overcome scale drift.

### **Main Contributions**

1. **Depth Estimation with CNNs**:
    - Uses a pre-trained CNN for single-image depth prediction.
    - only on keyframes
    - The network is trained on large-scale datasets to generalize well across different environments.
2. **Dense Depth Map Refinement**:
    - The estimated depth maps are refined using a probabilistic framework to improve consistency with geometric constraints.
3. **Optimization-based SLAM**:
    - Traditional SLAM techniques such as **pose graph optimization** and **bundle adjustment** are used to refine camera pose and depth.
    - CNN-based depth predictions act as a prior to regularize depth estimation and prevent scale drift.
4. **Real-time Performance**:
    - The method is designed to work in real time with computational efficiency improvements, making it suitable for real-world applications.

### Details

1. **Camera pose estimation**

![image.png](images/CNN-SLAM%20Real-time%20dense%20monocular%20SLAM%20with%20learn%201b471bdab3cf8032acd7f0ecedd731a4/image%202.png)

1. **CNN-based Depth Prediction and Semantic Segmentation**
    1. depth estimation on key frames
        - https://arxiv.org/abs/1606.00373
            
            **FCRN:** based on the extension of the Residual Network (ResNet) architecture [9] to a Fully Convolutional network.
            
            The first part of the architecture is based on ResNet-50 [9] and initialized with pre-trained weights on ImageNet [24]. 
            
            The second part of the architecture replaces the last pooling and fully connected layers originally presented in ResNet-50 with a sequence of residual upsampling blocks composed of a combination of unpooling and convolutional layers.
            
            ![image.png](images/CNN-SLAM%20Real-time%20dense%20monocular%20SLAM%20with%20learn%201b471bdab3cf8032acd7f0ecedd731a4/image%203.png)
            
            ![image.png](images/CNN-SLAM%20Real-time%20dense%20monocular%20SLAM%20with%20learn%201b471bdab3cf8032acd7f0ecedd731a4/image%204.png)
            
            [https://github.com/irolaina/FCRN-DepthPrediction](https://github.com/irolaina/FCRN-DepthPrediction)
            
2. Key-frame Creation andPose Graph Optimization
    
    depth adjustment：
    
    ![image.png](images/CNN-SLAM%20Real-time%20dense%20monocular%20SLAM%20with%20learn%201b471bdab3cf8032acd7f0ecedd731a4/image%205.png)
    
    uncertainty map：
    
    ![image.png](images/CNN-SLAM%20Real-time%20dense%20monocular%20SLAM%20with%20learn%201b471bdab3cf8032acd7f0ecedd731a4/image%206.png)
    
    propagated uncertainty:
    
    ![image.png](images/CNN-SLAM%20Real-time%20dense%20monocular%20SLAM%20with%20learn%201b471bdab3cf8032acd7f0ecedd731a4/image%207.png)
    
3. Frame-wise Depth Refinement The
    
    refine the depth map of the currently active key-frame based on the depth
    maps estimated at each new frame.
    
    ![image.png](images/CNN-SLAM%20Real-time%20dense%20monocular%20SLAM%20with%20learn%201b471bdab3cf8032acd7f0ecedd731a4/image%208.png)
    
    Importantly, since the key-frame is associated to a dense depth map thanks to the proposed CNN-based prediction, this process can be carried out densely, i.e. every element of the key-frame is refined, in contrast to [5] that only re- fines depth values along high gradient regions. Since the observed depths within low-textured regions tend to have a high-uncertainty (i.e., a high value in Ut), the proposed approach will naturally lead to a refined depth map where elements in proximity of high intensity gradients will be re- fined by the depth estimated at each frame, while elements within more and more low-textured regions will gradually hold the predicted depth value from the CNN, without being affected from uncertain depth observations.
    
4. Global Model and Semantic Label Fusion
    
    

### **Advantages**

- **Improved Depth Estimation**: Overcomes the limitations of purely geometric methods.
- **Reduced Scale Drift**: Uses learned depth to enhance pose estimation.
- **Dense and Accurate Mapping**: Generates a more complete 3D reconstruction compared to sparse SLAM methods.

### **Limitations**

- **Generalization Issues**: CNN-based depth prediction may not generalize well to unseen environments.
- **Dependency on Training Data**: The performance depends on the quality of the training dataset.
- **Computational Overhead**: While real-time, integrating CNNs into SLAM increases computational requirements.

**CNN-SLAM: Real-time Dense Monocular SLAM with Learned Depth Prediction** builds upon and enhances the **Direct Sparse Odometry (DSO)** method as its SLAM baseline.

### **Why DSO?**

- **Direct Method**: DSO directly optimizes photometric error instead of relying on keypoint features.
- **Sparse but Accurate**: It maintains a sparse set of points for optimization but provides high-precision tracking.
- **Minimizes Scale Drift**: Compared to feature-based methods like ORB-SLAM, DSO offers more robust pose estimation but still struggles with absolute scale.

### **How CNN-SLAM Enhances DSO**

1. **Depth Priors from CNNs**: CNN-SLAM integrates learned depth predictions as an additional constraint to refine the depth map.
2. **Dense Reconstruction**: Unlike the sparse depth maps in DSO, CNN-SLAM provides **dense depth estimation**.
3. **Improved Scale Estimation**: By using CNN-predicted depth, CNN-SLAM mitigates **scale drift**, a common issue in monocular SLAM.

Thus, CNN-SLAM extends DSO by incorporating **learning-based depth priors** to improve depth consistency and robustness in challenging scenarios.

![image.png](images/CNN-SLAM%20Real-time%20dense%20monocular%20SLAM%20with%20learn%201b471bdab3cf8032acd7f0ecedd731a4/image%209.png)

It proposes a monocular SLAM approach that fuses together **depth prediction via deep networks** and **direct monocular depth estimation** so to yield a dense scene reconstruction that is at the same time unambiguous in terms of absolute scale and robust in terms of tracking.

no pure rotation
