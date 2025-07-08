# runpod + VGGT

https://colmap.github.io/install.html

runpod [VGGT](https://www.notion.so/VGGT-1cb71bdab3cf801fb8cdddfb0e7282d5?pvs=21) 

git clone [https://github.com/facebookresearch/vggt.git](https://github.com/facebookresearch/vggt.git)
cd vggt
pip install -r requirements.txt

pip uninstall torchaudio
pip install torchaudio==2.3.1 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

viewer

pip install -r requirements_demo.txt

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
