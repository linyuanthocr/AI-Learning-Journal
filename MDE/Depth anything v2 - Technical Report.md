# Depth anything v2 - Technical Report

## 📘 Background

Monocular depth estimation is a fundamental computer vision task that aims to predict per-pixel depth from a single RGB image. 

data: synthetic images vs real data

![image.png](images/Depth%20anything%20v2%20-%20Technical%20Report%201c971bdab3cf80299210d7ef78a27fa4/image.png)

Two disadvantages of real labeled data. **1) Label noise**, i.e., inaccurate labels in depth maps. 

**2) Ignored details.** These real datasets often overlook certain details in their depth maps

![image.png](images/Depth%20anything%20v2%20-%20Technical%20Report%201c971bdab3cf80299210d7ef78a27fa4/image%201.png)

### 📊 Comparison: Synthetic vs. Real-World Data for Depth Estimation

| Aspect | Synthetic Data | Real-World Data |
| --- | --- | --- |
| **Scale** | ✅ Large-scale, easily generated | ❌ Limited due to costly annotation |
| **Ground Truth** | ✅ Perfect, noise-free depth maps | ❌ Noisy or sparse, inaccurate，sensor-dependent |
| **Domain Realism** | ❌ Unrealistic textures, lighting artifacts, too “clean” and “ordered” | ✅ Matches real deployment scenarios |
| **Diversity** | ✅ Customizable scene control | ❌ Limited by real-world collection |
| **Training Usage** | ✅ Pretraining, robustness enhancement | ✅ Finetuning, domain alignment |
| **Domain Shift Risk** | ⚠️ High when used alone | ✅ Low when targeting real environments |

Why synthetic data only?

Limitation 1. There exists **distribution shift** between synthetic and real images. Synthetic images are too “clean” in color and “ordered” in layout, while real images contain more randomness.

Limitation 2. Synthetic images have **restricted scene coverage.**  

Therefore, **synthetic-to-real transfer is non-trivial in MDE.**

## 🔍 Proposed Method

Depth Anything v2 builds on top of the original Depth Anything framework, introducing several improvements:

### 1. **Backbone (largest DINO v2 encoder: DINOv2-G)**

### 2. **Depth Decoder (DPT)**

### 3. **Training Strategy**

- train a reliable teacher model based on DINOv2-G purely on high-quality synthetic images.
- produce precise pseudo depth on large-scale unlabeled real images.
- train final student models on pseudo-labeled real images for robust generalization (we will show the synthetic images are not necessary in this step).

![image.png](images/Depth%20anything%20v2%20-%20Technical%20Report%201c971bdab3cf80299210d7ef78a27fa4/image%202.png)

![image.png](images/Depth%20anything%20v2%20-%20Technical%20Report%201c971bdab3cf80299210d7ef78a27fa4/image%203.png)

**LOSS function (screen shot of MiDaS)** 

![image.png](images/Depth%20anything%20v2%20-%20Technical%20Report%201c971bdab3cf80299210d7ef78a27fa4/image%204.png)

![image.png](images/Depth%20anything%20v2%20-%20Technical%20Report%201c971bdab3cf80299210d7ef78a27fa4/image%205.png)

images: 518*518：resizing the shorter size to 518 followed by a random crop.

The weight ratio of Lssi and Lgm is set as 1:2.

## 🔍 DA-2K

![image.png](images/Depth%20anything%20v2%20-%20Technical%20Report%201c971bdab3cf80299210d7ef78a27fa4/image%206.png)

## 🧪 Finetuning for Metric Depth

To adapt the model for **metric depth estimation**:

- **Supervised Finetuning**: Fine-tuned on datasets with metric ground truth (e.g., KITTI, NYUv2, DIML).
- **Loss Function**: Adds scale-aware L1/L2 losses in real-world units.
- **Depth Alignment**: Optionally uses GT-guided scale alignment or depth normalization.
- **Evaluation Metrics**: Uses standard metric depth metrics such as AbsRel, RMSE, δ thresholds.

## 📈 Results and Impact

Depth Anything v2 shows strong zero-shot and fine-tuned performance across multiple benchmarks, significantly narrowing the gap between monocular and stereo-based systems.

- **Zero-shot generalization**: Strong on indoor/outdoor datasets without finetuning.
- **Metric finetuning**: Competitive with state-of-the-art when adapted to metric scales.

## 📚 References

- [Depth Anything v2 GitHub](https://github.com/DepthAnything/Depth-Anything-V2)

https://arxiv.org/abs/2406.09414

### 🧪 Test

1. download source code
    
    git clone [https://github.com/DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
    cd Depth-Anything-V2/metric_depth
    pip install -r requirements.txt
    
2. download models
    
    https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth
    
3. run

```
python depth_to_pointcloud.py \
  --encoder vitl \
  --load-from checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
  --max-depth 20 \
  --img-path ~/Downloads/spiddy --outdir ~/Documents/3D/depth_anything_v2/metric_depth
```
