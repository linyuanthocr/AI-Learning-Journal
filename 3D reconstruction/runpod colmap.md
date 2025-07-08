# runpod + VGGT

https://colmap.github.io/install.html

# Setup Conda

```bash
#!/bin/bash

set -e  # 出错立即停止

echo "🚀 下载 Miniconda..."
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /workspace/Miniconda3.sh

echo "📦 安装 Miniconda 到 /workspace/miniconda3..."
bash /workspace/Miniconda3.sh -b -p /workspace/miniconda3

echo "🔧 配置环境变量..."
echo 'export PATH="/workspace/miniconda3/bin:$PATH"' >> ~/.bashrc
export PATH="/workspace/miniconda3/bin:$PATH"

echo "✅ 初始化 conda..."
/workspace/miniconda3/bin/conda init bash
source ~/.bashrc

echo "🧪 创建 Python 3.11 环境: vggt"
/workspace/miniconda3/bin/conda create -n vggt python=3.11 -y
source /workspace/miniconda3/bin/activate vggt
```

# Install VGGT

```bash
#!/bin/bash

set -e  # 一旦有命令失败就退出

source /workspace/miniconda3/bin/activate vggt

echo "📁 克隆 VGGT 仓库"
cd /workspace
git clone https://github.com/facebookresearch/vggt.git || echo "vggt repo already exists"
cd vggt

echo "📦 安装 requirements.txt"
pip install -r requirements.txt

echo "📦 安装 requirements_demo.txt"
pip install -r requirements_demo.txt

echo "🔧 开发者模式安装 VGGT"
pip install -e .

echo "📁 创建 HuggingFace 模型缓存目录"
mkdir -p /workspace/models/huggingface

echo "🌍 设置环境变量"
export HF_HOME=/workspace/models/huggingface
export TRANSFORMERS_CACHE=/workspace/models/huggingface

echo "✅ 所有安装完成！"

```

![image.png](images/runpod%20colmap%201cb71bdab3cf80c186c9eb85c894e561/image.png)
![image.png](images/runpod%20colmap%201cb71bdab3cf80c186c9eb85c894e561/image%201.png)
![image.png](images/runpod%20colmap%201cb71bdab3cf80c186c9eb85c894e561/image%202.png)
![image.png](images/runpod%20colmap%201cb71bdab3cf80c186c9eb85c894e561/image%203.png)
![image.png](images/runpod%20colmap%201cb71bdab3cf80c186c9eb85c894e561/image%204.png)
![image.png](images/runpod%20colmap%201cb71bdab3cf80c186c9eb85c894e561/image%205.png)
![image.png](images/runpod%20colmap%201cb71bdab3cf80c186c9eb85c894e561/image%206.png)
![image.png](images/runpod%20colmap%201cb71bdab3cf80c186c9eb85c894e561/image%207.png)
![image.png](images/runpod%20colmap%201cb71bdab3cf80c186c9eb85c894e561/image%208.png)


# VGGT+gsplat

### VGGT to colmap

in your SCENE_DIR, you should have a “images” folder which contains the images, and the following codes will generate a folder call “sparse” which contains the final colmap files you need for 3DGS training.

```bash
# Feedforward prediction only
python demo_colmap.py --scene_dir=/YOUR/SCENE_DIR/ 

# With bundle adjustment
python demo_colmap.py --scene_dir=/YOUR/SCENE_DIR/ --use_ba

# Run with bundle adjustment using reduced parameters for faster processing
# Reduces max_query_pts from 4096 (default) to 2048 and query_frame_num from 8 (default) to 5
# Trade-off: Faster execution but potentially less robust reconstruction in complex scenes (you may consider setting query_frame_num equal to your total number of images) 
# See demo_colmap.py for additional bundle adjustment configuration options
python demo_colmap.py --scene_dir=/YOUR/SCENE_DIR/ --use_ba --max_query_pts=2048 --query_frame_num=5
```
The reconstruction result (camera parameters and 3D points) will be automatically saved under /YOUR/SCENE_DIR/sparse/ in the COLMAP format, such as:
```
SCENE_DIR/
├── images/
└── sparse/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```
https://github.com/nerfstudio-project/gsplat

### runpod gsplat for VGGT results:

```bash
conda create -n gsplat
conda activate gsplat
pip install git+https://github.com/nerfstudio-project/gsplat.gitgit clone https://github.com/nerfstudio-project/gsplat.git
cd gsplat
cd examples
pip install -r requirements.txt
```

in examples folder, run simple_trainer and set the correct scene and result dir

```bash
python simple_trainer.py  default --data_factor 1 --data_dir /YOUR/SCENE_DIR/ --result_dir /YOUR/RESULT_DIR/
```
