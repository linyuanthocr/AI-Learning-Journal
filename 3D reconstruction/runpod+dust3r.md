# runpod+dust3r

# pod template:

runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

![image.png](images/runpod+dust3r%201d371bdab3cf800998a4f1f3170203ac/image.png)

# Install Miniconda

---

### ✅ **Step-by-Step Guide: Install Miniconda on Ubuntu 22.04**

### 1. **Download the Miniconda installer**

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

```

### 2. **Run the installer**

```bash
bash Miniconda3-latest-Linux-x86_64.sh

```

- Press `Enter` to continue.
- Type `yes` to accept the license.
- Choose the install location (default is usually fine).
- Say `yes` to initialize Miniconda.

### 3. **Restart your shell (important)**

```bash
source ~/.bashrc

```

If you're using zsh:

```bash
source ~/.zshrc

```

---

### 🎉 **Verify Installation**

```bash
conda --version

```

You should see something like:

```
conda 24.1.2

```

---

# Install Dust3r

1. Clone DUSt3R.

```bash
git clone --recursive https://github.com/naver/dust3r
cd dust3r
# if you have already cloned dust3r:
# git submodule update --init --recursive
```

1. Create the environment, here we show an example using conda.

```bash
conda create -n dust3r python=3.11 cmake=3.14.0
conda activate dust3r
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
# Optional: you can also install additional packages to:
# - add support for HEIC images
# - add pyrender, used to render depthmap in some datasets preprocessing
# - add required packages for visloc.py
pip install -r requirements_optional.txt
```

1. Optional, compile the cuda kernels for RoPE (as in CroCo v2).

```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

![image.png](images/runpod+dust3r%201d371bdab3cf800998a4f1f3170203ac/image%201.png)

**Checkpoints**

You can obtain the checkpoints by two ways:

1. You can use our huggingface_hub integration: the models will be downloaded automatically.
2. Otherwise, We provide several pre-trained models:

| Modelname | Training resolutions | Head | Encoder | Decoder |
| --- | --- | --- | --- | --- |
| [`DUSt3R_ViTLarge_BaseDecoder_224_linear.pth`](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth) | 224x224 | Linear | ViT-L | ViT-B |
| [`DUSt3R_ViTLarge_BaseDecoder_512_linear.pth`](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth) | 512x384, 512x336, 512x288, 512x256, 512x160 | Linear | ViT-L | ViT-B |
| [`DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth) | 512x384, 512x336, 512x288, 512x256, 512x160 | DPT | ViT-L | ViT-B |

You can check the hyperparameters we used to train these models in the [section: Our Hyperparameters](https://github.com/naver/dust3r#our-hyperparameters)

To download a specific model, for example `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`:

```
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/
```

For the checkpoints, make sure to agree to the license of all the public training datasets and base checkpoints we used, in addition to CC-BY-NC-SA 4.0. Again, see [section: Our Hyperparameters](https://github.com/naver/dust3r#our-hyperparameters) for details.

# Dust3r_demo_no_viewer

[v3_demo.py](dust3r_demo_nogradio.py)

Run

```bash
python v3_demo.py \
  --image_path ../samples/spidy \
  --output_dir ../outputs/spidy \
  --model_name checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --image_size 512 \
  --as_pointcloud \
  --clean_depth \
  --transparent_cams

```

RESULTS

![snapshot04.png](images/runpod+dust3r%201d371bdab3cf800998a4f1f3170203ac/snapshot04.png)

![snapshot03.png](images/runpod+dust3r%201d371bdab3cf800998a4f1f3170203ac/snapshot03.png)

![snapshot05.png](images/runpod+dust3r%201d371bdab3cf800998a4f1f3170203ac/snapshot05.png)

![snapshot06.png](images/runpod+dust3r%201d371bdab3cf800998a4f1f3170203ac/snapshot06.png)
