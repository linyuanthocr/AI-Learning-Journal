# Diffusion model — SDXL

[arxiv.org](https://arxiv.org/pdf/2307.01952)

[zhuanlan.zhihu.com](https://zhuanlan.zhihu.com/p/642496862)

之前的文章[文生图模型之Stable Diffusion](https://zhuanlan.zhihu.com/p/617134893)已经介绍了比较火的文生图模型Stable Diffusion，近期Stability AI又发布了新的升级版本[SDXL](https://link.zhihu.com/?target=https%3A//stability.ai/blog/sdxl-09-stable-diffusion)。目前SDXL的代码、模型以及技术报告已经全部开源：

- 官方代码：[https://github.com/Stability-AI/generative-models](https://link.zhihu.com/?target=https%3A//github.com/Stability-AI/generative-models)
- 模型权重：[https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9](https://link.zhihu.com/?target=https%3A//huggingface.co/stabilityai/stable-diffusion-xl-base-0.9)
- 技术报告：[SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2307.01952)

而且SDXL也已经集成在了huggingface的diffusers库中：[diffusers/pipelines/stable_diffusion_xl](https://link.zhihu.com/?target=https%3A//github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/stable_diffusion_xl)。SDXL和之前的版本一样也是采用latent diffusion架构，但SDXL相比之前的版本SD 1.x和SD 2.x有明显的提升，下面是SDXL和之前SD 1.5和SD 2.1的一些直观对比图：

![Untitled](Diffusion%20model%20%E2%80%94%20SDXL%20034fb5c11e2b4b82b3e5931c50a86a6a/Untitled.png)

可以看到SDXL无论是在文本理解还是在生成图像质量上，相比之前的版本均有比较大的提升。SDXL性能的提升主要归功于以下几点的改进：

- **SDXL的模型参数增大为2.3B，这几乎上原来模型的3倍，而且SDXL采用了两个CLIP text encoder来编码文本特征；**
- **SDXL采用了额外的条件注入来改善训练过程中的数据处理问题，而且最后也采用了多尺度的微调；**
- **SDXL级联了一个细化模型来提升图像的生成质量。**

这篇文章我们将结合SDXL的代码来具体讲解上述的改进技巧。

### **模型架构上的优化**

SDXL和之前的版本也是基于**latent diffusion架构**，对于latent diffusion，首先会采用一个autoencoder模型来图像压缩为latent，然后扩散模型用来生成latent，生成的latent可以通过autoencoder的decoder来重建出图像。SDXL的autoencoder依然采用KL-f8，但是并没有采用之前的autoencoder，而是**基于同样的架构采用了更大的batch size（256 vs 9）重新训练，同时采用了EMA**。重新训练的VAE模型（尽管和VAE有区别，大家往往习惯称VAE）相比之前的模型，其重建性能有一定的提升，性能对比如下所示：

![Untitled](Diffusion%20model%20%E2%80%94%20SDXL%20034fb5c11e2b4b82b3e5931c50a86a6a/Untitled%201.png)

这里要注意的是上表中的三个VAE模型其实模型结构是完全一样，其中SD-VAE 2.x只是在SD-VAE 1.x的基础上重新微调了decoder部分，但是encoder权重是相同的，所以两者的latent分布是一样的，两个VAE模型是都可以用在SD 1.x和SD 2.x上的。但是SDXL-VAE是完全重新训练的，它的latent分布发生了改变，你**不可以将SDXL-VAE应用在SD 1.x和SD 2.x上**。在将latent送入扩散模型之前，我们要对latent进行缩放来使得latent的标准差尽量为1，由于权重发生了改变，所以**SDXL-VAE的缩放系数也和之前不同，之前的版本采用的缩放系数为0.18215，而SDXL-VAE的缩放系数为0.13025**。SDXL-VAE的权重也已经单独上传到huggingface上（[https://huggingface.co/stabilityai/sdxl-vae](https://link.zhihu.com/?target=https%3A//huggingface.co/stabilityai/sdxl-vae)），一个要注意的点是SDXL-VAE采用float16会出现溢出（具体见[这里](https://link.zhihu.com/?target=https%3A//github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py%23L786)），**必须要使用float32来进行推理**，但是之前的版本使用float16大部分情况都是可以的。VAE的重建能力对SD生成的图像质量还是比较重要的，SD生成的图像容易出现小物体畸变，这往往是由于VAE导致的，SDXL-VAE相比SD-VAE 2.x的提升其实比较微弱，所以也不会大幅度缓解之前的畸变问题。

SDXL相比之前的版本，一个最大的变化采用了更大的UNet，下表为SDXL和之前的SD的具体对比，之前的SD的UNet参数量小于1B，但是**SDXL的UNet参数量达到了2.6B，比之前的版本足足大了3倍**。

![Untitled](Diffusion%20model%20%E2%80%94%20SDXL%20034fb5c11e2b4b82b3e5931c50a86a6a/Untitled%202.png)

下面我们来重点看一下SDXL是如何扩增UNet参数的，SDXL的UNet模型结构如下图所示：

![Untitled](Diffusion%20model%20%E2%80%94%20SDXL%20034fb5c11e2b4b82b3e5931c50a86a6a/Untitled%203.png)

相比之前的SD，SDXL的第一个stage采用的是普通的DownBlock2D，而不是采用基于attention的CrossAttnDownBlock2D，这个主要是为了计算效率，因为SDXL最后是直接生成1024x1024分辨率的图像，对应的latent大小为128x128x4，如果第一个stage就使用了attention（包含self-attention），所需要的显存和计算量都是比较大的。另外一个变化是SDXL只用了3个stage，这意味着只进行了两次2x下采样，而之前的SD使用4个stage，包含3个2x下采样。SDXL的网络宽度（这里的网络宽度是指的是特征channels）相比之前的版本并没有改变，3个stage的特征channels分别是320、640和1280。**SDXL参数量的增加主要是使用了更多的transformer blocks**，在之前的版本，每个包含attention的block只使用一个transformer block（self-attention -> cross-attention -> ffn），但是SDXL中stage2和stage3的两个CrossAttnDownBlock2D模块中的transformer block数量分别设置为2和10，并且中间的MidBlock2DCrossAttn的transformer blocks数量也设置为10（和最后一个stage保持一样）。可以看到SDXL的UNet在空间维度最小的特征上使用数量较多的transformer block，这是计算效率最高的。

SDXL的另外一个变动是text encoder，SD 1.x采用的text encoder是123M的OpenAI CLIP ViT-L/14，而SD 2.x将text encoder升级为354M的OpenCLIP ViT-H/14。SDXL更进一步，不仅采用了更大的[OpenCLIP ViT-bigG](https://link.zhihu.com/?target=https%3A//laion.ai/blog/giant-openclip/)（参数量为694M），而且同时也用了OpenAI CLIP ViT-L/14，这里是分别提取两个text encoder的倒数第二层特征，其中OpenCLIP ViT-bigG的特征维度为1280，而CLIP ViT-L/14的特征维度是768，两个特征concat在一起总的特征维度大小是2048，这也就是SDXL的context dim。OpenCLIP ViT-bigG相比OpenCLIP ViT-H/14，在性能上有一定的提升，其中在ImageNet上zero-shot性能为80.1%。强大的text encoder对于文生图模型的文本理解能力是至关重要的。

![Untitled](Diffusion%20model%20%E2%80%94%20SDXL%20034fb5c11e2b4b82b3e5931c50a86a6a/Untitled%204.png)

这里有一个处理细节是提取了OpenCLIP ViT-bigG的pooled text embedding（用于CLIP对比学习所使用的特征），将其映射到time embedding的维度并与之相加。这种特征嵌入方式在强度上并不如cross attention，只是作为一种辅助。

经过上述调整，SDXL的UNet总参数量为2.6B。SDXL只是UNet变化了，而扩散模型的设置是和原来的SD一样，都采用1000步的DDPM，noise scheduler也保持没动，训练损失是采用基于预测noise的𝐿simple。

### **额外的条件注入**

SDXL的第二个优化点采用了额外的条件注入来解决训练过程中数据处理问题，这里包括两种条件注入方式，它们分别解决训练过程中**数据利用效率和图像裁剪问题**。

首先我们来看第一个问题，SD的训练往往是先在256x256上预训练，然后在512x512上继续训练。当使用256x256尺寸训练时，要过滤掉那些宽度和高度小于256的图像，采用512x512尺寸训练时也同样只用512x512尺寸以上的图像。由于需要过滤数据，这就导致实际可用的训练样本减少了，要知道训练数据量对大模型的性能影响是比较大。下图展示了SDXL预训练数据的图像尺寸分布，可以看到如果要过滤256一下的图像，就其实丢掉了39%的训练样本。

一种直接的解决方案是采用一个超分模型先对数据进行预处理，但是目前超分模型并不是完美的，还是会出现一些artifacts（对于pixel diffusion模型比如Imagen，往往是采用级联的模型，64x64的base模型加上两个超分模型，其中base模型的数据利用效率是比较高的，但是可能的风险是超分模型也可能会出现artifacts）。SDXL提出了一种简单的方案来解决这个问题，那就是**将图像的原始尺寸（width和height）作为条件嵌入UNet模型中，这相当于让模型学到了图像分辨率参数，在训练过程中，我们可以不过滤数据直接resize图像，在推理时，我们只需要送入目标分辨率而保证生成的图像质量。**图像原始尺寸嵌入的实现也比较简单，和timesteps的嵌入一样，先将width和height用傅立叶特征编码进行编码，然后将特征concat在一起加在time embedding上。下图展示了采用这种方案得到的512x512模型当送入不同的size时的生成图像对比，可以看到模型已经学到了识别图像分辨率，当输入低分辨率时，生成的图像比较模糊，但是当提升size时，图像质量逐渐提升。

![Untitled](Diffusion%20model%20%E2%80%94%20SDXL%20034fb5c11e2b4b82b3e5931c50a86a6a/Untitled%205.png)

第二个问题是训练过程中的**图像裁剪**问题，目前文生图模型预训练往往采用固定图像尺寸，这就需要对原始图像进行预处理，这个处理流程一般是先将图像的最短边resize到目标尺寸，然后沿着图像的最长边进行裁剪（random crop或者center crop）。但是图像裁剪往往会导致图像出现缺失问题，比如下图采用center crop导致人物的头和脚缺失了，这也直接导致模型容易生成缺损的图像。

为了解决这个问题，**SDXL也将训练过程中裁剪的左上定点坐标作为额外的条件注入到UNet中**，这个注入方式可以采用和图像原始尺寸一样的方式，即通过傅立叶编码并加在time embedding上。在推理时，我们只需要将这个坐标设置为(0, 0)就可以得到物体居中的图像（此时图像相当于没有裁剪）。下图展示了采用不同的crop坐标的生成图像对比，可以看到(0, 0)坐标可以生成物体居中而无缺失的图像，采用其它的坐标就会出现有裁剪效应的图像。

SDXL在训练过程中，可以将两种条件注入（size and crop conditioning）结合在一起使用，训练数据的处理流程和之前是一样的，只是要额外保存图像的**原始width和height**以及图像**crop时的左上定点坐标top和left**，具体的流程如下所示：

![Untitled](Diffusion%20model%20%E2%80%94%20SDXL%20034fb5c11e2b4b82b3e5931c50a86a6a/Untitled%206.png)

这里我们简单总结一下，SDXL总共增加了4个额外的条件注入到UNet，它们分别是pooled text embedding，original size，crop top-left coord和target size。对于后面三个条件，它们可以像timestep一样采用傅立叶编码得到特征，然后我们这些特征和pooled text embedding拼接在一起，最终得到维度为2816（1280+256*2*3）的特征。我们将这个特征采用两个线性层映射到和time embedding一样的维度，然后加在time embedding上即可，具体的实现代码如下所示：

```python
import math
from einops import rearrange
import torch

batch_size =16
# channel dimension of pooled output of text encoder (s)
pooled_dim = 1280
adm_in_channels = 2816
time_embed_dim = 1280

def fourier_embedding(inputs, outdim=256, max_period=10000):
    """
    Classical sinusoidal timestep embedding
    as commonly used in diffusion models
    : param inputs : batch of integer scalars shape [b ,]
    : param outdim : embedding dimension
    : param max_period : max freq added
    : return : batch of embeddings of shape [b, outdim ]
    """
    half = outdim // 2
    freqs = torch.exp(
        -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
    ).to(device=inputs.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding

def cat_along_channel_dim(x: torch.Tensor,) -> torch.Tensor:
    if x.ndim == 1:
        x = x[... , None]
 assert x . ndim == 2
 b, d_in = x.shape
    x = rearrange(x, "b din -> (b din)")
    # fourier fn adds additional dimension
    emb = fourier_embedding(x)
    d_f = emb.shape[-1]
    emb = rearrange(emb, "(b din) df -> b (din df)",
                     b=b, din=d_in, df=d_f)
 return emb

def concat_embeddings(
    # batch of size and crop conditioning cf. Sec. 3.2
    c_size: torch.Tensor,
    c_crop: torch.Tensor,
    # batch of target size conditioning cf. Sec. 3.3
    c_tgt_size: torch.Tensor ,
    # final output of text encoders after pooling cf. Sec . 3.1
    c_pooled_txt: torch.Tensor,
) -> torch.Tensor:
    # fourier feature for size conditioning
    c_size_emb = cat_along_channel_dim(c_size)
 # fourier feature for size conditioning
 c_crop_emb = cat_along_channel_dim(c_crop)
 # fourier feature for size conditioning
 c_tgt_size_emb = cat_along_channel_dim(c_tgt_size)
 return torch.cat([c_pooled_txt, c_size_emb, c_crop_emb, c_tgt_size_emd], dim=1)

# the concatenated output is mapped to the same
# channel dimension than the noise level conditioning
# and added to that conditioning before being fed to the unet
adm_proj = torch.nn.Sequential(
    torch.nn.Linear(adm_in_channels, time_embed_dim),
    torch.nn.SiLU(),
    torch.nn.Linear(time_embed_dim, time_embed_dim)
)

# simulating c_size and c_crop as in Sec. 3.2
c_size = torch.zeros((batch_size, 2)).long()
c_crop = torch.zeros((batch_size, 2)).long ()
# simulating c_tgt_size and pooled text encoder output as in Sec. 3.3
c_tgt_size = torch.zeros((batch_size, 2)).long()
c_pooled = torch.zeros((batch_size, pooled_dim)).long()
 
# get concatenated embedding
c_concat = concat_embeddings(c_size, c_crop, c_tgt_size, c_pooled)
# mapped to the same channel dimension with time_emb
adm_emb = adm_proj(c_concat)
```

### **细化模型**

SDXL的另外一个优化点是级联了一个**细化模型**（**refiner model**）来进一步提升图像质量，如下图所示：

![Untitled](Diffusion%20model%20%E2%80%94%20SDXL%20034fb5c11e2b4b82b3e5931c50a86a6a/Untitled%207.png)

这里第一个模型我们称为base model，上述我们讲的其实就是SDXL-base model，第二个模型是refiner model，它是进一步在base model生成的图像基础上提升图像的细节。refiner model是和base model采用同样VAE的一个latent diffusion model，但是它只在使用较低的noise level进行训练（只在前200 timesteps上），在推理时，我们**只使用refiner model的图生图能力**。对于一个prompt，我们首先用base model生成latent，然后我们给这个latent加一定的噪音（采用扩散过程），并使用refiner model进行去噪。经过这样一个重新加噪再去噪的过程，图像的局部细节会有一定的提升.

级联refiner model其实相当于一种模型集成，这种集成策略也早已经应用在文生图中，比如NVIDA在[eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2211.01324)就提出了集成不同的扩散模型来提升生成质量。另外采用SD的图生图来提升质量其实也早已经被应用了，比如社区工具[Stable Diffusion web UI](https://link.zhihu.com/?target=https%3A//github.com/AUTOMATIC1111/stable-diffusion-webui)的[high res fix](https://link.zhihu.com/?target=https%3A//github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/6509)就是基于图生图来实现的（结合超分模型）。

refiner model和base model在结构上有一定的不同，其UNet的结构如下图所示，refiner model采用4个stage，第一个stage也是采用没有attention的DownBlock2D，网络的特征维度采用384，而base model是320。另外，refiner model的attention模块中transformer block数量均设置为4。refiner model的参数量为2.3B，略小于base model。

![Untitled](Diffusion%20model%20%E2%80%94%20SDXL%20034fb5c11e2b4b82b3e5931c50a86a6a/Untitled%208.png)

另外refiner model的text encoder只使用了OpenCLIP ViT-bigG，也是提取倒数第二层特征以及pooled text embed。与base model一样，refiner model也使用了size and crop conditioning，除此之外还增加了图像的艺术评分[aesthetic-score](https://link.zhihu.com/?target=https%3A//github.com/christophschuhmann/improved-aesthetic-predictor)作为条件，处理方式和之前一样。refiner model应该没有采用多尺度微调，所以没有引入target size作为条件（refiner model只是用来图生图，它可以直接适应各种尺度）。