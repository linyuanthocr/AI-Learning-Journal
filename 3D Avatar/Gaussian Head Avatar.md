# Gaussian Head Avatar

![image.png](images/Gaussian%20Head%20Avatar%201a771bdab3cf801aae03d5b74aec9e04/image.png)

![image.png](images/Gaussian%20Head%20Avatar%201a771bdab3cf801aae03d5b74aec9e04/image%201.png)

![image.png](images/Gaussian%20Head%20Avatar%201a771bdab3cf801aae03d5b74aec9e04/image%202.png)

输入：多视角多个视频序列

1. 图像去除背景，针对每张图像：用3DMM进行初始人脸构建+估计3D feature loc+expression参数估计。
2. 构建标准高斯脸X。每个顶点有128维特征（决定颜色）+3D rotation+1D scale+1D opacity 分别用F，R，S，A表示

![image.png](images/Gaussian%20Head%20Avatar%201a771bdab3cf801aae03d5b74aec9e04/image%203.png)

3.利用基于MLP+表情条件构建动态生成器。训练时不光学习phi，还要优化标准脸X。

![image.png](images/Gaussian%20Head%20Avatar%201a771bdab3cf801aae03d5b74aec9e04/image%204.png)

1. 训练过程：
    1. gaussian脸在表情+pose变化时的新位置X’
        
        ![image.png](images/Gaussian%20Head%20Avatar%201a771bdab3cf801aae03d5b74aec9e04/image%205.png)
        
        assume that the **Gaussian points closer to 3D land-marks are more affected by the expression coefficients** and less affected by the head pose. These landmarks initialized with 3DMM, optimized in initialization
        
        ![image.png](images/Gaussian%20Head%20Avatar%201a771bdab3cf801aae03d5b74aec9e04/image%206.png)
        
    
    b. Color C’:
    
    ![image.png](images/Gaussian%20Head%20Avatar%201a771bdab3cf801aae03d5b74aec9e04/548a4e62-93e5-48e2-9184-4ebd5896ec01.png)
    
    c. Rotation, scale and opacity:
    
    ![image.png](images/Gaussian%20Head%20Avatar%201a771bdab3cf801aae03d5b74aec9e04/image%207.png)
    

### 训练具体步骤

1. 预处理：如上述第1步所述
2. 每一步迭代过程中：
    1. 利用公式2，生成3D 高斯头初始化
    2. 利用公式1，生成512*512*32的输入图像（X:3,C:9,R:4,S:3,A:1, camera parameters mu:12）
3.
