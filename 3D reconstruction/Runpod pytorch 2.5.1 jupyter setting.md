# Runpod + pytorch 2.5.1+jupyter setting

Background: setting your own pytorch version on Runpod

![image.png](images/Runpod%20pytorch%20jupyter%20setting/image.png)

在 Runpod **Ubuntu 22.04** 上安装 **Miniconda** 的完整步骤如下：

```python
bash install_miniconda_torch.sh
```

### 🔧 `install_miniconda_torch.sh`

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

echo "🧪 创建 Python 3.10 环境: torch251"
/workspace/miniconda3/bin/conda create -n torch251 python=3.10 -y
source /workspace/miniconda3/bin/activate torch251

echo "🧠 安装 PyTorch 2.5.1 + CUDA 12.1..."
conda install -n torch251 pytorch==2.5.1 torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo "🧹 清理缓存释放空间..."
conda clean -a -y

echo "✅ 安装完成！请执行："
echo "    source ~/.bashrc"
echo "    conda activate torch251"
echo "    python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"

```

## ✅ 最终效果

- Miniconda 安装在 `/workspace/miniconda3`
- 环境 `torch251` 使用 Python 3.10 + PyTorch 2.5.1 (CUDA 12.1)
- `PATH` 已加入 `.bashrc`
- 不污染容器层（减轻容器使用率）

---

## ✅ 正确方式：让内置 Jupyter 能识别你的 Conda 环境

### ✳️ 1. 在你的环境中安装 `ipykernel`

```bash
conda activate torch251
pip install ipykernel

```

---

### ✳️ 2. 注册为新的内核

```bash
python -m ipykernel install --user --name=torch251 --display-name "Python (torch251)"

```

- `-name` 是 Jupyter 用的内部名
- `-display-name` 是你在 JupyterLab UI 中看到的名字

---

### ✅ 3. 在 Jupyter 中选择你的环境

接下来，在 **RunPod 的 Jupyter UI** 页面中：

- 点击右上角 ➕ 打开新 Notebook
- 在 kernel 列表里选择 `Python (torch251)`

---

## 📌 补充说明

- **无需**重新启动 Jupyter 或重新安装 Jupyter 本体
- 如果你想删除这个 kernel，运行：
    
    ```bash
    jupyter kernelspec uninstall torch251
    
    ```
    

---

To run the Kaggle competition download command on RunPod, follow these steps:

---

### 1. Launch a RunPod Instance

1. **Log in to RunPod**: Visit [runpod.io](https://www.runpod.io/) and sign in to your account.([Kaggle Solutions](https://kaggle.curtischong.me/tools/runpod.io?utm_source=chatgpt.com))
2. **Start a Pod**: Navigate to the "Secure Cloud" section and launch a new pod. Choose an image that includes Python and pip, such as the PyTorch 2.4 + CUDA 12.4 image .([RunPod - The Cloud Built for AI](https://www.runpod.io/articles/guides/pytorch-2.4-cuda-12.4?utm_source=chatgpt.com))
3. **Access the Terminal**: Once the pod is running, open the terminal interface via SSH or the web terminal provided by RunPod.

---

### 2. Install the Kaggle CLI

In the terminal, install the Kaggle command-line tool:

```bash
pip install kaggle

```

---

### 3. Set Up Kaggle API Credentials

1. **Obtain Your API Token**:
    - Go to your Kaggle account settings: [https://www.kaggle.com/account](https://www.kaggle.com/account).
    - Scroll down to the "API" section and click on "Create New API Token". This will download a file named `kaggle.json`.
2. **Upload `kaggle.json` to RunPod**:
    - Use the RunPod file upload interface or SCP to transfer the `kaggle.json` file to your pod.
    - Place the file in the `/root/.kaggle/` directory. If the `.kaggle` directory doesn't exist, create it:
        
        ```bash
        mkdir -p /root/.kaggle
        mv kaggle.json /root/.kaggle/
        chmod 600 /root/.kaggle/kaggle.json
        
        ```
        

---

### 4. Download the Competition Dataset

With the Kaggle CLI configured, you can now download the dataset:

```bash
kaggle competitions download -c image-matching-challenge-2025

```

This command will download all files associated with the "image-matching-challenge-2025" competition into your current working directory.

---

### 5. Extract the Dataset (If Necessary)

If the downloaded files are compressed (e.g., ZIP files), extract them using:

```bash
unzip image-matching-challenge-2025.zip

```

Replace the filename with the actual name of the downloaded ZIP file if it's different.

---

### Additional Tips

- **Persistent Storage**: RunPod's Secure Cloud offers persistent storage. Ensure you're working within a directory that retains data between sessions if you plan to revisit your work later.([Thunder Compute](https://www.thundercompute.com/blog/top-google-colab-alternatives-for-cheap-gpu-powered-deep-learning-in-2025?utm_source=chatgpt.com))
- **Data Management**: For large datasets, consider using RunPod's network storage options to manage your data efficiently .

```markdown
kaggle competitions download -c image-matching-challenge-2025

#!/bin/bash
kaggle models instances versions download metaresearch/dinov2/pyTorch/base/1
#!/bin/bash
kaggle models instances versions download oldufo/lightglue/pyTorch/aliked/1
#!/bin/bash
kaggle models instances versions download oldufo/aliked/pyTorch/aliked-n16/1
#!/bin/bash
kaggle datasets download oldufo/imc2024-packages-lightglue-rerun-kornia
#!/bin/bash
kaggle datasets download eduardtrulls/imc25-utils
```

torch251 kaggle

好的！你想把这段设置 Kaggle API key 的操作：

```bash
mkdir -p /root/.kaggle
mv kaggle.json /root/.kaggle/
chmod 600 /root/.kaggle/kaggle.json

```

改为适用于 **RunPod 的 Conda 环境 torch251（路径：`/workspace/envs/torch251`）** 中。

---

### ✅ 推荐做法：使用 **当前用户目录**，不使用 `/root`

因为 `/root/.kaggle` 是 root 用户专属，在 RunPod 容器中你可能不是用真正的 root 登录，或者权限受限。

Kaggle 官方建议的配置路径是：

```
~/.kaggle/kaggle.json

```

---

### ✅ 修改版本（适配 Conda 环境、非 root 用户）

```bash
mkdir -p /workspace/envs/torch251/.kaggle
mv kaggle.json /workspace/envs/torch251/.kaggle/kaggle.json
chmod 600 /workspace/envs/torch251/.kaggle/kaggle.json

```

---

### ✅ 设置环境变量（推荐添加到激活脚本）

你需要让 Kaggle CLI 知道你的 token 位置：

```bash
export KAGGLE_CONFIG_DIR=/workspace/envs/torch251/.kaggle

```

你可以把这行加到激活脚本里：

```bash
echo 'export KAGGLE_CONFIG_DIR=/workspace/envs/torch251/.kaggle' >> /workspace/envs/torch251/etc/conda/activate.d/env_vars.sh
chmod +x /workspace/envs/torch251/etc/conda/activate.d/env_vars.sh

```

---

### ✅ 完整方案总结：

```bash
# 1. 放 key 到合适目录
mkdir -p /workspace/envs/torch251/.kaggle
mv kaggle.json /workspace/envs/torch251/.kaggle/kaggle.json
chmod 600 /workspace/envs/torch251/.kaggle/kaggle.json

# 2. 设置环境变量以让 kaggle CLI 找到它
export KAGGLE_CONFIG_DIR=/workspace/envs/torch251/.kaggle

# 可选：永久写入激活脚本中
mkdir -p /workspace/envs/torch251/etc/conda/activate.d
echo 'export KAGGLE_CONFIG_DIR=/workspace/envs/torch251/.kaggle' > /workspace/envs/torch251/etc/conda/activate.d/env_vars.sh
chmod +x /workspace/envs/torch251/etc/conda/activate.d/env_vars.sh

```

---

完成后你就可以正常运行命令，比如：

```bash
kaggle competitions download -c rsna-2024-lumbar-spine

```

需要我帮你封装成一键 `.sh` 初始化脚本吗？
