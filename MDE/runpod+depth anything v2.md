# runpod+depth anything v2

1. GPU select: 1 x RTX 6000 Ada
16 vCPU 188 GB RAM
2. runpod template **RunPod Pytorch 2.1 (**runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04)
3. jupyter terminal download and install depth anything v2 (default dir: /workspace):
    
    ```bash
    git clone https://github.com/DepthAnything/Depth-Anything-V2
    cd Depth-Anything-V2
    pip install -r requirements.txt
    ```
    
4. install depth anything v2 metric related requirement
    
    ```bash
    cd /workspace/Depth-Anything-V2/metric_depth
    pip install -r requirements.txt
    ```
    
5. download model
    
    ```bash
    mkdir -p checkpoints
    cd /workspace/Depth-Anything-V2/metric_depth/checkpoints
    wget "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true" -O depth_anything_v2_metric_hypersim_vitl.pth
    ```
    
6. update dataset
7. run in terminal

```bash
cd /workspace/Depth-Anything-V2/metric_depth
python run.py \
  --encoder vitl \
  --load-from /workspace/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth \
  --max-depth 20 \
  --img-path /workspace/spiddy --outdir /workspace/outputs/spiddy --save-numpy \
  --input-size 518
```

![WechatIMG6.png](images/runpod+depth%20anything%20v2%201ca71bdab3cf805ba068e510ff4ed516/WechatIMG6.png)

![WechatIMG26.png](images/runpod+depth%20anything%20v2%201ca71bdab3cf805ba068e510ff4ed516/WechatIMG26.png)

![WechatIMG32.png](images/runpod+depth%20anything%20v2%201ca71bdab3cf805ba068e510ff4ed516/WechatIMG32.png)
