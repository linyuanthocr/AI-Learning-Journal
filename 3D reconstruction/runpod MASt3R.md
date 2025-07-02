# runpod + MASt3R

# Download the Miniconda installer

wget [https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) -O [miniconda.sh](http://miniconda.sh/)

# Run the installer

bash [miniconda.sh](http://miniconda.sh/) -b -p $HOME/miniconda

# Initialize conda

$HOME/miniconda/bin/conda init

# Reload shell (or restart the terminal)

source ~/.bashrc

# Verify installation

conda --version

**Installation**

```bash
conda create -n mast3r-slam python=3.11
conda activate mast3r-slam

```

Check the system's CUDA version with nvcc

```bash
nvcc --version

```

Install pytorch with **matching** CUDA version following:

```bash
# CUDA 11.8
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# CUDA 12.4
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

```

Clone the repo and install the dependencies.

```bash
git clone https://github.com/rmurai0610/MASt3R-SLAM.git --recursive
cd MASt3R-SLAM/

# if you've clone the repo without --recursive run
# git submodule update --init --recursive

pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install --no-build-isolation -e .

# Optionally install torchcodec for faster mp4 loading
pip install torchcodec==0.1

```

Setup the checkpoints for MASt3R and retrieval. The license for the checkpoints and more information on the datasets used is written [here](https://github.com/naver/mast3r/blob/mast3r_sfm/CHECKPOINTS_NOTICE).

```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
```

run MASt3R

```
python main.py --dataset <path/to/video>.mp4 --config config/base.yaml

```