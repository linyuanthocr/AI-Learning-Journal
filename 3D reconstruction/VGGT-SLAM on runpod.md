# VGGT-SLAM on runpod

[https://github.com/MIT-SPARK/VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM)

https://arxiv.org/abs/2505.12549

### install conda & initialization

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
/workspace/miniconda3/bin/conda create -n vggt-slam  python=3.11 -y
source /workspace/miniconda3/bin/activate vggt-slam
```

### Install VGGT-SLAM

**Clone VGGT-SLAM:**

```bash
git clone https://github.com/MIT-SPARK/VGGT-SLAM
cd VGGT-SLAM
```

**Install dependencies:**

```bash
pip install open3d numpy
conda install -c conda-forge boost
pip install pycolmap
```

**Make the setup script executable and run it**

This step will automatically download all 3rd party packages including VGGT. More details on the license for VGGT can be found [here](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt).

```
chmod +x setup.sh

```

```
./setup.sh

```

### Run VGGT-SLAM

**Quick Start**

run `python main.py --image_folder /path/to/image/folder --max_loops 1 --vis_map` replacing the image path with your folder of images. This will create a visualization in viser which shows the incremental construction of the map.

As an example, we provide a folder of test images in office_loop.zip which will generate the following map. Using the default parameters will result in a single loop closure towards the end of the trajectory. Unzip the folder and set its path as the arguments for --image_folder

[](https://github.com/MIT-SPARK/VGGT-SLAM/raw/main/assets/office-loop-figure)

*Running in the default SL(4) mode on this folder will show significant drift in the projective degrees of freedom before the loop closure, and the drift will be corrected after the loop closure. You may notice drift in other scenes as well if the system goes too long without a loop closure. We are actively working on an upgraded VGGT-SLAM that will have significantly reduced drift and other major updates so stay tuned!*

**Collecting Custom Data**

To quickly collect a test on a custom dataset, you can record a trajectory with a cell phone and convert the MOV file to a folder of images with:

```
mkdir <desired_location>/img_folder

```

And then, run the command below:

```
ffmpeg -i /path/to/video.MOV -vf "fps=10" <desired_location>/img_folder/frame_%04d.jpg

```

**Adjusting Parameters**

See main.py or run `--help` from main.py to view all parameters. We use SL(4) mode by default, and Sim(3) mode can be enabled with `--use_sim3`. Sim(3) mode will generally have less drift than SL(4) but will not always be sufficient for alignment (see paper for in depth discussion on the advantages of SL(4)).

### output pose ply files (cannot use viewer on runpod)

1. uncomment .pcd output line in main.py

![image.png](images/VGGT-SLAM%20on%20runpod%2022b71bdab3cf80e784bec1a3f9b391a4/image.png)

1. run the slam:

```bash
python main.py --image_folder spiddy --max_loops 1 --vis_map --log_results
```

in this way you will get: poses.txt and poses_points.pcd

1. turn pcd to ply

```python
import open3d as o3d
import numpy as np

# === 输入输出路径 ===
pcd_path = "poses_points.pcd"
ply_path = "poses_points.ply"
ply_downsample_path = "poses_points_downsampled.ply"

# === 加载 .pcd 点云
pcd = o3d.io.read_point_cloud(pcd_path)
print(f"📌 原始点数: {len(pcd.points)}")

# # === 直接写出为 .ply
# o3d.io.write_point_cloud(ply_path, pcd)
# print(f"✅ 已保存为：{ply_path}")

# === 设置下采样 voxel 大小（单位：米或坐标单位）
voxel_size = 0.003  # 例如每 1cm 保留一个点，你可调大试试 0.05 或 0.1

# === 体素下采样
down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
print(f"✅ 下采样后点数: {len(down_pcd.points)}")

# === 写出为 .ply 文件
o3d.io.write_point_cloud(ply_downsample_path, down_pcd)
print(f"✅ 已保存为：{ply_downsample_path}")
```

1. view ply in MashLab

![snapshot04.png](images/VGGT-SLAM%20on%20runpod%2022b71bdab3cf80e784bec1a3f9b391a4/snapshot04.png)

![snapshot02.png](images/VGGT-SLAM%20on%20runpod%2022b71bdab3cf80e784bec1a3f9b391a4/snapshot02.png)

![snapshot01.png](images/VGGT-SLAM%20on%20runpod%2022b71bdab3cf80e784bec1a3f9b391a4/snapshot01.png)

![snapshot00.png](images/VGGT-SLAM%20on%20runpod%2022b71bdab3cf80e784bec1a3f9b391a4/snapshot00.png)
