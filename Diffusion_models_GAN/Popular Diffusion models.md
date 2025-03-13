# Popular Diffusion models

# DALLE

[unCLIp: Dalle](https://www.notion.so/unCLIp-Dalle-323fede6597443af8d479fb9e5f0f0ef?pvs=21) 

[How DALL-E 2 Actually Works](https://www.assemblyai.com/blog/how-dall-e-2-actually-works/)

# GLIDE

[从DDPM到GLIDE：基于扩散模型的图像生成算法进展](https://zhuanlan.zhihu.com/p/449284962?utm_psn=1730808131335135232)

# Imagen

[Imagen](https://www.notion.so/Imagen-faa04e5bea3d488585f829f89a07e0df?pvs=21) 

# Latent Diffusion Model

[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

[文生图模型之Stable Diffusion](https://zhuanlan.zhihu.com/p/617134893?utm_psn=1730824401908662272)

## 简介

这个工作，常规的扩散模型是基于pixel的生成模型，而Latent Diffusion是基于latent的生成模型，它先采用一个autoencoder将图像压缩到latent空间，然后用扩散模型来生成图像的latents，最后送入autoencoder的decoder模块就可以得到生成的图像。

![Untitled](images/Popular%20Diffusion%20models%20eb9e858a18874bee9cafc7e276e9e701/Untitled.png)

SD模型的主体结构如下图所示，主要包括三个模型：

- **autoencoder**：encoder将图像压缩到latent空间，而decoder将latent解码为图像；
- **CLIP text encoder**：提取输入text的text embeddings，通过cross attention方式送入扩散模型的UNet中作为condition；
- **UNet**：扩散模型的主体，用来实现文本引导下的latent生成。

![Untitled](images/Popular%20Diffusion%20models%20eb9e858a18874bee9cafc7e276e9e701/Untitled.webp)

## Autoencoder

![Untitled](images/Popular%20Diffusion%20models%20eb9e858a18874bee9cafc7e276e9e701/Untitled%201.png)

[latent diffusion loss](https://github.com/CompVis/latent-diffusion/tree/main/ldm/modules/losses)

![Untitled](images/Popular%20Diffusion%20models%20eb9e858a18874bee9cafc7e276e9e701/Untitled%202.png)

![Untitled](images/Popular%20Diffusion%20models%20eb9e858a18874bee9cafc7e276e9e701/Untitled%203.png)

```python
import torch
from diffusers import AutoencoderKL
import numpy as np
from PIL import Image

#加载模型: autoencoder可以通过SD权重指定subfolder来单独加载
autoencoder = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
autoencoder.to("cuda", dtype=torch.float16)

# 读取图像并预处理
raw_image = Image.open("boy.png").convert("RGB").resize((256, 256))
image = np.array(raw_image).astype(np.float32) / 127.5 - 1.0
image = image[None].transpose(0, 3, 1, 2)
image = torch.from_numpy(image)

# 压缩图像为latent并重建
with torch.inference_mode():
    latent = autoencoder.encode(image.to("cuda", dtype=torch.float16)).latent_dist.sample()
    rec_image = autoencoder.decode(latent).sample
    rec_image = (rec_image / 2 + 0.5).clamp(0, 1)
    rec_image = rec_image.cpu().permute(0, 2, 3, 1).numpy()
    rec_image = (rec_image * 255).round().astype("uint8")
    rec_image = Image.fromarray(rec_image[0])
rec_image
```

![Untitled](images/Popular%20Diffusion%20models%20eb9e858a18874bee9cafc7e276e9e701/Untitled%204.png)

## **CLIP text encoder**

SD**采用CLIP text encoder来对输入text提取text embeddings**，具体的是采用目前OpenAI所开源的最大CLIP模型：[clip-vit-large-patch14](https://link.zhihu.com/?target=https%3A//huggingface.co/openai/clip-vit-large-patch14)，这个CLIP的text encoder是一个**transformer模型（只有encoder模块）**：层数为12，特征维度为768，模型参数大小是123M。对于输入text，送入CLIP text encoder后得到最后的hidden states（即最后一个transformer block得到的特征），其特征维度大小为77x768（77是token的数量），**这个细粒度的text embeddings将以cross attention的方式送入UNet中**。在transofmers库中，可以如下使用CLIP text encoder：

```python
from transformers import CLIPTextModel, CLIPTokenizer

text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder").to("cuda")
# text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# 对输入的text进行tokenize，得到对应的token ids
prompt = "a photograph of an astronaut riding a horse"
text_input_ids = text_tokenizer(
    prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt"
).input_ids

# 将token ids送入text model得到77x768的特征
text_embeddings = text_encoder(text_input_ids.to("cuda"))[0]
```

值得注意的是，这里的tokenizer最大长度为77（CLIP训练时所采用的设置），当输入text的tokens数量超过77后，将进行截断，如果不足则进行paddings，这样将保证无论输入任何长度的文本（甚至是空文本）都得到77x768大小的特征。 在训练SD的过程中，**CLIP text encoder模型是冻结的**。在早期的工作中，比如OpenAI的[GLIDE](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2112.10741)和latent diffusion中的LDM均采用一个随机初始化的tranformer模型来提取text的特征，但是最新的工作都是采用预训练好的text model。比如谷歌的Imagen采用纯文本模型T5 encoder来提出文本特征，而SD则采用CLIP text encoder，预训练好的模型往往已经在大规模数据集上进行了训练，它们要比直接采用一个从零训练好的模型要好。

## UNet

SD的扩散模型是一个860M的UNet，其主要结构如下图所示（这里以输入的latent为64x64x4维度为例），其中encoder部分包括3个CrossAttnDownBlock2D模块和1个DownBlock2D模块，而decoder部分包括1个UpBlock2D模块和3个CrossAttnUpBlock2D模块，中间还有一个UNetMidBlock2DCrossAttn模块。encoder和decoder两个部分是完全对应的，中间存在skip connection。注意3个CrossAttnDownBlock2D模块最后均有一个2x的downsample操作，而DownBlock2D模块是不包含下采样的。

![Untitled](images/Popular%20Diffusion%20models%20eb9e858a18874bee9cafc7e276e9e701/Untitled%205.png)

其中CrossAttnDownBlock2D模块的主要结构如下图所示，text condition将通过CrossAttention模块嵌入进来，此时Attention的query是UNet的中间特征，而key和value则是text embeddings。 CrossAttnUpBlock2D模块和CrossAttnDownBlock2D模块是一致的，但是就是总层数为3。

![Untitled](images/Popular%20Diffusion%20models%20eb9e858a18874bee9cafc7e276e9e701/Untitled%206.png)

### UNet code

```python
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)
```

```python
class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)
```

```python
class Model(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, use_timestep=True, use_linear_attn=False, attn_type="vanilla"):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t=None, context=None):
        #assert x.shape[2] == x.shape[3] == self.resolution
        if context is not None:
            # assume aligned context, cat along channel axis
            x = torch.cat((x, context), dim=1)
        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        return self.conv_out.weight
```

![Untitled](Popular%20Diffusion%20models%20eb9e858a18874bee9cafc7e276e9e701/Untitled%207.png)

```python

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F

# 加载autoencoder
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
# 加载text encoder
text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
# 初始化UNet
unet = UNet2DConditionModel(**model_config) # model_config为模型参数配置
# 定义scheduler
noise_scheduler = DDPMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
)

# 冻结vae和text_encoder
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

opt = torch.optim.AdamW(unet.parameters(), lr=1e-4)

for step, batch in enumerate(train_dataloader):
    with torch.no_grad():
        # 将image转到latent空间
        latents = vae.encode(batch["image"]).latent_dist.sample()
        latents = latents * vae.config.scaling_factor # rescaling latents
        # 提取text embeddings
        text_input_ids = text_tokenizer(
            batch["text"],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
  ).input_ids
  text_embeddings = text_encoder(text_input_ids)[0]
    
    # 随机采样噪音
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    # 随机采样timestep
    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
    timesteps = timesteps.long()

    # 将noise添加到latent上，即扩散过程
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # 预测noise并计算loss
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

 opt.step()
    opt.zero_grad()
```

注意的是SD的noise scheduler虽然也是采用一个1000步长的scheduler，但是不是linear的，而是scaled linear，具体的计算如下所示：

```python
betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
```

在训练条件扩散模型时，往往会采用**Classifier-Free Guidance**（这里简称为CFG），所谓的CFG简单来说就是在训练条件扩散模型的同时也训练一个无条件的扩散模型，同时在采样阶段将条件控制下预测的噪音和无条件下的预测噪音组合在一起来确定最终的噪音，具体的计算公式如下所示：

![Untitled](images/Popular%20Diffusion%20models%20eb9e858a18874bee9cafc7e276e9e701/Untitled%208.png)

这里的$\omega$为**guidance scale**，当$\omega$越大时，condition 起的作用越大，即生成的图像其更和输入文本一致。CFG的具体实现非常简单，在训练过程中，我们只需要**以一定的概率（比如10%）随机drop掉text**即可，这里我们可以将text置为空字符串（前面说过此时依然能够提取text embeddings）。这里并没有介绍CLF背后的技术原理，感兴趣的可以阅读CFG的论文

[Classifier-Free Diffusion Guidance](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2207.12598) 以及guided diffusion的论文 [Diffusion Models Beat GANs on Image Synthesis](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2105.05233) **CFG对于提升条件扩散模型的图像生成效果是至关重要的**。

[SD 应用](images/Popular%20Diffusion%20models%20eb9e858a18874bee9cafc7e276e9e701/SD%20%E5%BA%94%E7%94%A8%202eae7a95ea4e489d9859a2f942bc2bc9.md)

## SDXL

[zhuanlan.zhihu.com](https://zhuanlan.zhihu.com/p/642496862)

[Diffusion model — SDXL](images/Popular%20Diffusion%20models%20eb9e858a18874bee9cafc7e276e9e701/Diffusion%20model%20%E2%80%94%20SDXL%20034fb5c11e2b4b82b3e5931c50a86a6a.md)

[**Stable diffusion library**](https://huggingface.co/blog/stable_diffusion)

- [**The Annotated Diffusion Model**](https://huggingface.co/blog/annotated-diffusion)
- [**Getting started with 🧨 Diffusers**](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb)

[**https://huggingface.co/docs/diffusers/index**](https://huggingface.co/docs/diffusers/index)

[Stable_Diffusion_Diagrams_V2.pdf](images/Popular%20Diffusion%20models%20eb9e858a18874bee9cafc7e276e9e701/Stable_Diffusion_Diagrams_V2.pdf)

# Scalable Diffusion Models with Transformers (DiT)

[](https://arxiv.org/pdf/2212.09748.pdf)

We explore a new class of diffusion models based on the transformer architecture. We train latent diffusion models of images, replacing the commonly-used U-Net backbone with a transformer that operates on latent patches. 

**Diffusion Transformers, or DiTs**

## **Patchify**

The input to DiT is a spatial representation z (for 256 × 256 × 3 images, z has shape 32 × 32 × 4). The first layer of DiT is “patchify,” which converts the spatial input into a sequence of T tokens, each of dimension d, by **linearly embedding** each patch in the input. Following patchify, we apply standard ViT **frequency-based positional embeddings (the sine-cosine version) to all input tokens**. The number of tokens T created by patchify is determined by the patch size hyperparameter p. As shown in Figure 4, halving p will quadruple T, and thus at least quadruple total transformer Gflops. Although it has a significant impact on Gflops, note that changing p has no meaningful impact on downstream parameter counts. We add p = 2, 4, 8 to the DiT design space.

![Untitled](images/Popular%20Diffusion%20models%20eb9e858a18874bee9cafc7e276e9e701/Untitled%209.png)

## DiT block design

![Untitled](images/Popular%20Diffusion%20models%20eb9e858a18874bee9cafc7e276e9e701/Untitled%2010.png)

Following patchify, the input tokens are processed by a sequence of transformer blocks. In addition to noised image inputs, diffusion models sometimes process additional conditional information such as **noise timesteps t, class labels c**, natural language, etc. We explore **four variants of transformer blocks** that process conditional inputs differently. The designs introduce small, but important, modifications to the standard ViT block design. The designs of all blocks are shown in Figure 3. 

![Untitled](images/Popular%20Diffusion%20models%20eb9e858a18874bee9cafc7e276e9e701/Untitled%2011.png)

### In-context conditioning

append the vector embeddings of **t and c** as **two additional tokens** in the input sequence, treating them no differently from the image tokens. This is similar to cls tokens in ViTs, and it allows us to use standard ViT blocks without modification. After the final block, we remove the conditioning tokens from the sequence. 

### Cross-attention block

We concatenate the embeddings of t and c into a length-two sequence, separate from the image token sequence. The transformer block is modified to include an **additional multi-head cross- attention layer following the multi-head self-attention block**, similar to the original design from Vaswani et al. [60], and also similar to the one used by LDM for conditioning on class labels.

### Adaptive layer norm (adaLN) block

Following the widespread usage of **adaptive normalization layers** [40] in GANs [2, 28] and diffusion models with U- Net backbones [9], we explore replacing standard layer norm layers in transformer blocks with adaptive layer norm (adaLN). Rather than directly learn dimension- wise **scale and shift parameters** γ and β, we **regress** them from the sum of the embedding vectors of t and c. It is also the only conditioning mechanism that is restricted to **apply the same function to all tokens**.

[](https://arxiv.org/pdf/1911.07013.pdf)

![Untitled](images/Popular%20Diffusion%20models%20eb9e858a18874bee9cafc7e276e9e701/Untitled%2012.png)

### adaLN-Zero block

Zero-initializing the final convolutional layer in each block prior to any residual connections. In addition to regressing γ and β, we also regress dimension- wise scaling parameters α that are applied immediately prior to any residual connections within the DiT block. We **initialize the MLP to output the zero-vector for all α; this initializes the full DiT block as the identity function.** 

### Model size

We apply a sequence of N DiT blocks, each operating at the hidden dimension size d. Following ViT, we use standard transformer configs that jointly scale N, d and attention heads

![Untitled](images/Popular%20Diffusion%20models%20eb9e858a18874bee9cafc7e276e9e701/Untitled%2013.png)

### Transformer decoder.

After the final DiT block, we need to decode our sequence of image tokens into an **output noise prediction and an output diagonal covariance prediction**. Both of these outputs have **shape equal** to the original spatial input. We use a standard **linear decoder** to do this; we apply the final layer norm (adaptive if using adaLN) and linearly decode each token into a p×p×2C tensor, where C is the number of channels in the spatial input to DiT. Finally,  we rearrange the decoded tokens into their original spatial layout to get the predicted noise and covariance. The complete DiT design space we explore is **patch size, transformer block architecture and model size**

For the rest of the paper, all models will use **adaLN-Zero DiT blocks**

[](https://github.com/facebookresearch/DiT/blob/main/models.py)

[DiT model](images/Popular%20Diffusion%20models%20eb9e858a18874bee9cafc7e276e9e701/DiT%20model%20a73a071c51544374997ae8db29ed92fb.md)

[Diffusion model：DiT, transformer only](images/Popular%20Diffusion%20models%20eb9e858a18874bee9cafc7e276e9e701/Diffusion%20model%EF%BC%9ADiT,%20transformer%20only%208a22b30972f343d0befe0f0a373d8387.md)
