# Stable diffusion 3

paper link：

[arxiv.org](https://arxiv.org/pdf/2403.03206)

文生图SD3：

[zhuanlan.zhihu.com](https://zhuanlan.zhihu.com/p/686273242)

• Our new Multimodal Diffusion Transformer (MMDiT) architecture uses separate sets of weights for image and language representations, which improves text understanding and spelling capabilities compared to previous versions of Stable Diffusion.

## **改进的RF**

SD3相比之前的SD一个最大的变化是采用**Rectified Flow**来作为生成模型，Rectified Flow在[Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2209.03003)被首先提出，但其实也有同期的工作比如[Flow Matching for Generative Modeling](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2210.02747)提出了类似的想法。这里和SD3的论文一样，首先将基于**Flow Matching**来介绍RF，然后再介绍SD3在RF上的具体改进。

### **Flow Matching**

**Flow Matching（FM）**是建立在[continuous normalizing flows](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1806.07366)的基础上，这里将生成模型定义为一个**常微分方程（ODE）**：

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled.png)

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%201.png)

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%202.png)

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%203.png)

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%204.png)

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%205.png)

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%206.png)

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%207.png)

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%208.png)

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%209.png)

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2010.png)

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2011.png)

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2012.png)

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2013.png)

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2014.png)

**可以看到rf/lognorm(0.00, 1.00)是综合rank最高的**，而且在5 steps和50 steps下也可以取得较好的rank。这里所采用的lognorm(0.00, 1.00)的时间采样方法也恰好是偏向中间时间步的，这说明对中间时间步加权是重要且有效的。这里也可以看到未改进的rf效果上反而是不如LDM所采用的eps/linear，而且经典的eps/linear的rank也仅次于几个改进的rf。

下表展示了不同的模型在25 steps下具体的CLIP score和FID，rf/lognorm(0.00, 1.00)两个数据集均表现不错，而经典的eps/linear(噪声是线性的, linear noise schedule)其实也不差。

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2015.png)

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2016.png)

可以看到rf模型在steps比较小时展现比较明显的优势，说明rf模型可以减少推理阶段的采样步数。当steps增加时，rf不如eps/linear，但是改进后的rf/lognorm(0.00, 1.00)依然能够超过eps/linear。

**总结：RF模型推理高效，但是通过改进时间采样方法对中间时间步加权能进一步提升效果，这里基于lognorm(0.00, 1.00)的采样方法从实验看是最优的。**

## **多模态DiT**

SD3除了采用改进的RF，另外一个重要的改进就是采用了一个多模态DiT。多模态DiT的一个核心对图像的latent tokens和文本tokens拼接在一起，并采用两套独立的权重处理，但是在attention时统一处理。整个架构图如下所示：

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2017.png)

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2018.png)

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2019.png)

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2020.png)

### **文本编码器**

SD3的text encoder包含3个预训练好的模型：

- [CLIP ViT-L](https://link.zhihu.com/?target=https%3A//huggingface.co/openai/clip-vit-large-patch14)：参数量约124M
- [OpenCLIP ViT-bigG](https://link.zhihu.com/?target=https%3A//huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)：参数量约695M
- [T5-XXL encoder](https://link.zhihu.com/?target=https%3A//huggingface.co/google/t5-v1_1-xxl)：参数量约4.7B

SD 1.x模型的text encoder使用CLIP ViT-L，SD 2.x模型的text encoder采用OpenCLIP ViT-H，而SDXL的text encoder使用CLIP ViT-L + OpenCLIP ViT-bigG。这次SD3更上一个台阶，加上了一个更大的T5-XXL encoder。谷歌的[Imagen](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2205.11487)最早使用T5-XXL encoder作为文生图模型的text encoder，并证明预训练好的纯文本模型可以实现更好的文本理解能力，后面的工作，如NVIDIA的[eDiff-I](https://link.zhihu.com/?target=https%3A//research.nvidia.com/labs/dir/eDiff-I/)和Meta的[Emu](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2309.15807)采用T5-XXL encoder + CLIP作为text encoder，OpenAI的DALL-E 3也采用T5-XXL encoder。SD3加入T5-XXL encoder也是模型在文本理解能力特别是文字渲染上提升的一个关键。

具体地，SD3总共提取两个层面的特征。 首先提取两个CLIP text encoder的pooled embedding，它们是文本的全局语义特征，维度大小分别是768和1280，两个embedding拼接在一起得到2048的embedding，然后经过一个MLP网络之后和timestep embedding相加。 然后是文本细粒度特征。这里也先分别提取两个CLIP模型的倒数第二层的特征，拼接在一起可以得到77x2048维度的CLIP text embeddings；同样地也从T5-XXL encoder提取最后一层的特征T5 text embeddings，维度大小是77x4096（这里也限制token长度为77）。然后对CLIP text embeddings使用zero-padding得到和T5 text embeddings同维度的特征。最后，将padding后的CLIP text embeddings和T5 text embeddings在token维度上拼接在一起，得到154x4096大小的混合text embeddings。text embeddings将通过一个linear层映射到与图像latent的patch embeddings同维度大小，并和patch embeddings拼接在一起送入MM-DiT中。

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2021.png)

### **MM-DiT**

MM-DiT和DiT一样也是处理图像latent空间，这里先对图像的latent转成patches，这里的patch size=2x2，和DiT的默认配置是一样的。patch embedding再加上positional embedding送入transformer中。 这里的重点是如何处理前面说的文本特征。对于CLIP pooled embedding可以直接和timestep embedding加在一起，并像DiT中所设计的adaLN-Zero一样将特征插入transformer block。

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2022.png)

具体的实现代码如下所示：

```python
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

```

对于序列的text embeddings，常规的处理方式是增加cross attention层来处理，其中text embeddings作为attention的keys和values，比如SD的UNet以及[PIXART-α](https://link.zhihu.com/?target=https%3A//pixart-alpha.github.io/)（基于DiT）。但是SD3是直接将text embeddings和patch embeddings拼在一起处理，这样不需要额外引入cross-attention。由于text和image属于两个不同的模态，这里采用两套独立的参数来处理，即所有transformer层的学习参数是不共享的，但是共用一个self-attention来实现特征的交互。这等价于采用两个transformer模型来处理文本和图像，但在attention层连接，所以这是一个多模态模型，称之为MM-DiT。

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2023.png)

MM-DiT和之前文生图模型的一个区别是文本特征不再只是作为一个条件，而是和图像特征同等对待处理。论文中也基于CC12M数据集将MM-DiT和其它架构做了对比实验，这里对比的模型有DiT（这里的DiT是指的不引入cross-attention，直接将text tokens和patches拼接，但只有一套参数），CrossDiT（额外引入cross-attention），UViT（UNet和transformer混合架构），还有3套参数的MM-DiT（CLIP text tokens，T5-XXL text tokens和patches各一套参数）。不同架构的模型表现如下所示：

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2024.png)

### **QK-Normalization**

为了提升混合精度训练的稳定性，MM-DiT的self-attention层还采用了QK-Normalization。当模型变大，而且在高分辨率图像上训练时，attention层的attention-logit（Q和K的矩阵乘）会变得不稳定，导致训练出现NAN。这里的解决方案是采用[RMSNorm](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1910.07467)（简化版LayerNorm）对attention的Q和K进行归一化。

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2025.png)

### **变尺度位置编码**

MM-DiT的位置编码和ViT一样采用2d的frequency embeddings（两个1d frequency embeddings进行concat）。SD3先在256x256尺寸下预训练，但最终会在以1024x1024为中心的多尺度上微调，这就需要MM-DiT的位置编码需要支持变尺度。SD3采用的解决方案是**插值+扩展**。

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2026.png)

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2027.png)

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2028.png)

### **timestep schedule的shift**

对高分辨率的图像，如果采用和低分辨率图像的一样的noise schedule，会出现对图像的破坏不够的情况，如下图所示（图源自[On the Importance of Noise Scheduling for Diffusion Models](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2301.10972)）：

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2029.png)

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2030.png)

### **模型scaling**

transformer一个比较大的优势是有好的scaling能力：当增大模型带来性能的稳定提升。论文中也选择了不同规模大小的MM-DiT进行实验，不同大小的网络深度分别是15，18，21，30，38，其中最大的模型参数量为8B。结论是MM-DiT同样表现了比较好的scaling能力，当模型变大后，性能稳步提升，如下图所示：

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2031.png)

这里的另外一个结论是validation loss可以作为一个很好的模型性能的衡量指标，它和文生图模型的一些评测指标如[CompBench](https://link.zhihu.com/?target=https%3A//karine-h.github.io/T2I-CompBench/)和[GenEval](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2310.11513)，以及人类偏好是正相关的。而且从目前的实验结果来看，还没有看到出现性能的饱和，这意味着继续增大模型，依然有可能继续提升。 下图展示了三个不同大小的模型生成图像的差异，可以看到大模型确实是质量最好的

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2032.png)

而且更大的模型不仅性能更好，而且生成时可以用较少的采样步数，比如当步数为5步时，大模型的性能下降要比小模型要低。

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2033.png)

## **实现细节**

这部分简单介绍一下SD3的一些实现细节，包括训练数据的处理以及训练参数等。

### **预训练数据处理**

预训练数据集的大小和来源是没有的，但是预训练数据会进行一些筛选，包括：

1. 色情内容：使用NSFW检测模型来过滤。
2. 图像美学：使用评分系统移除预测分数较低的图像。
3. 重复内容：基于聚类的去重方法来移除训练数据中重复的图像，防止模型直接复制训练数据集中图像。（这部分策略附录部分很详细）

### **图像caption**

和DALL-E 3一样，这里也对训练数据集中的图像生成高质量caption，这里使用的模型是多模态大模型[CogVLM](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2311.03079)。训练过程中，使用50%的原始caption和50%的合成caption，使用合成caption能够提升模型性能，如下表所示。

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2034.png)

### **预计算图像和文本特征**

为了减少训练过程中所需显存，这里预先计算好图像经过autoencoder编码得到的latent，以及文本对应的text embedding，特别是T5，可以节省接近20B的显存。同时预先计算好特征，也会节省一部分时间。

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2035.png)

但是预计算特征也不是没有代价的，首先是图像就不能做数据增强，好在文生图模型训练一般不太需要数据增强，其次需要一定的存储空间，而且加载特征也需要时间。预计算特征其实就是空间换时间。

### **Classifier-Free Guidance**

训练过程需要对文本进行一定的drop来实现Classifier-Free Guidance，这里是三个text encoder各以46.4%的比例单独drop，这也意味着text完全drop的比例为(46.4%)^3≈10%。

三个text encoder独立drop的一个好处是推理时可以灵活使用text encoder。比如，我们可以去掉比较吃显存的T5模型，只保留两个CLIP text encoder，实验发现这并不会影响视觉美感（没有T5的胜率为50%），并且只会导致文本遵循度略有下降（胜率为46%），这种情况包括文本提示词包含高度详细的场景描述或大量文字。然而，如果想生成文字，还是加上T5，没有T5的胜率只有38%。下面是一些具体的例子：

### **DPO**

SD3最后基于DPO来进一步提升性能，DPO相比RLHF的一个优势不需要单独训练一个reward模型，而且直接基于成对的比较数据训练。DPO目前已经成功应用在文生图上：[Diffusion Model Alignment Using Direct Preference Optimization](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2311.12908)。SD3这里没有finetune整个网络，而是基于rank=128的LoRA，经过DPO后，图像生成质量有一定的提升，如下所示：

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2036.png)

[Paper Review: Diffusion Model Alignment Using Direct Preference Optimization](https://artgor.medium.com/paper-review-diffusion-model-alignment-using-direct-preference-optimization-cb6e75c0da0b)

## **性能评测**

性能评测包括定量评测和人工评测。

### **定量评测**

定量评测基于GenEval，SD3和其它模型的对比如下所示，可以看到最大的模型在经过DPO后超过DALL-E 3。

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2037.png)

### **人工评测**

人工评测包括三个方面：

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2038.png)

评测结果如下所示，这里对比的模型有SOTA的模型：MJ-V6，Ideogram-V1.0，DALL-E 3，在文字生成方面，SD3基本大幅赢过其它模型（和Ideogram-V1.0相差上下），在图像质量和文本提示词遵循方面也和SOTA模型不相上下。

![Untitled](Stable%20diffusion%203%20c73bab85d94944edbb991761fe6b6d06/Untitled%2039.png)

## **小结**

SD3可以说是集大成者，基本上把业界最好的或者最成熟的方案都用上了，比如RF和DiT，以及DPO等等。SD3的正式发布，也基本宣告文生图进入transformer时代了，现在的模型才是8B，未来更大的模型也定会出现。