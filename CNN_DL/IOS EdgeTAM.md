# IOS EdgeTAM

[EdgeTAM code](https://github.com/facebookresearch/EdgeTAM)

[EdgeTAM paper](https://arxiv.org/abs/2501.07256)

SAM 2 builds on the Segment Anything Model (SAM), extending its capabilities from images to videos using a memory bank mechanism, achieving strong performance in video segmentation. To make SAM 2 more efficient and mobile-friendly, we introduce EdgeTAM, targeting not just the image encoder but also the memory attention blocks, which our benchmarks show as a major latency bottleneck. EdgeTAM uses a lightweight 2D Spatial Perceiver with fixed learnable queries to encode frame-level memories while preserving spatial structure through global and patch-level queries. A distillation strategy further boosts performance without added inference cost. EdgeTAM reaches 87.7, 70.0, 72.3, and 71.7 J&F on DAVIS 2017, MOSE, SA-V val, and SA-V test, respectively, running at 16 FPS on iPhone 15 Pro Max.

# Install EdgeTAM on mac pro

```bash
# Step 1: Development Environment Setup

# First, check your system
echo "System check:"
echo "Python version: $(python3 --version)"
echo "Operating system: $(uname -s)"
echo "Architecture: $(uname -m)"

# Create conda environment (if you have conda installed)
# If you don't have conda, install it first from: https://docs.conda.io/en/latest/miniconda.html
conda create -n edgetam python=3.10 -y
conda activate edgetam

conda install pytorch torchvision torchaudio -c pytorch -c conda-forge

# 如果不好用，每次新开terminal后的标准流程：
# conda deactivate  # 完全退出conda环境
# conda activate edgetam  # 干净地进入目标环境

# Test PyTorch installation
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'MPS available: {torch.backends.mps.is_available()}')"

# Step 2. Intall EdgeTAM
# Clone EdgeTAM repository
git clone https://github.com/facebookresearch/EdgeTAM.git
cd EdgeTAM

# Install EdgeTAM
pip install -e .

# Install additional dependencies for notebooks and iOS conversion
pip install -e ".[notebooks]"
pip install coremltools
pip install torch-model-archiver

# Verify the installation directory structure
echo "EdgeTAM directory structure:"
ls -la

# Verify coremltools installation
python -c "import coremltools as ct; print(f'Core ML Tools version: {ct.__version__}')"

# Step 3. Test EdgeTAM model

# Test that EdgeTAM loads correctly
# 安装timm和其他可能缺少的依赖
pip install timm
pip install hydra-core
pip install omegaconf

# 查看更详细的错误信息
HYDRA_FULL_ERROR=1 python -c "
import torch
import os

# 强制使用CPU
torch.cuda.is_available = lambda: False

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = './checkpoints/edgetam.pt'
model_cfg = 'configs/edgetam.yaml'

try:
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device='cpu'))
    print('✅ EdgeTAM model loaded successfully on CPU!')
    print(f'Model device: {next(predictor.model.parameters()).device}')
except Exception as e:
    print(f'❌ Error loading EdgeTAM: {e}')
"
```

### Test on Mac pro

```bash
pip install gradio

python gradio_app.py
```

![image.png](images/iphone%20EdgeTAM%2022971bdab3cf80ffaaa0d899c867551f/image.png)

### to be continued
