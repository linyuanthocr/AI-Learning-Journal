# Diffusion model：DiT, transformer only

扩散模型大部分是采用**UNet架构**来进行建模，UNet可以实现输出和输入一样维度，所以天然适合扩散模型。扩散模型使用的UNet除了包含基于残差的卷积模块，同时也往往采用self-attention。自从ViT之后，transformer架构已经大量应用在图像任务上，随着扩散模型的流行，也已经有工作尝试采用transformer架构来对扩散模型建模，这篇文章我们将介绍Meta的工作**DiT**：[**Scalable Diffusion Models with Transformers**](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2212.09748)，它是**完全基于transformer架构的扩散模型**，这个工作不仅将transformer成功应用在扩散模型，还探究了**transformer架构在扩散模型上的scalability能力**，**其中最大的模型DiT-XL/2在ImageNet 256x256的类别条件生成上达到了SOTA（FID为2.27）**。

![Untitled](Diffusion%20model%EF%BC%9ADiT,%20transformer%20only%208a22b30972f343d0befe0f0a373d8387/Untitled.png)

在介绍DiT模型架构之前，我们先来看一下DiT所采用的扩散模型。 首先，DiT并没有采用常规的pixel diffusion，而是**采用了latent diffusion架构**，这也是Stable Diffusion所采用的架构。latent diffusion采用一个autoencoder来将图像压缩为低维度的latent，扩散模型用来生成latent，然后再采用autoencoder来重建出图像。DiT采用的autoencoder是SD所使用的KL-f8，对于256x256x3的图像，其压缩得到的latent大小为32x32x4，这就降低了扩散模型的计算量（后面我们会看到这将减少transformer的token数量）。另外，这里扩散过程的nosie scheduler采用简单的linear scheduler（timesteps=1000，beta_start=0.0001，beta_end=0.02），这个和SD是不同的。 其次，DiT所使用的扩散模型沿用了OpenAI的[**Improved DDPM**](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2102.09672)，相比原始DDPM一个重要的变化是不再采用固定的方差，而是**采用网络来预测方差**。在DDPM中，生成过程的分布采用一个参数化的高斯分布来建模：

![Untitled](Diffusion%20model%EF%BC%9ADiT,%20transformer%20only%208a22b30972f343d0befe0f0a373d8387/Untitled%201.png)

上面介绍完了DiT所采用的扩散模型设置，然后我们来介绍DiT所设计的transformer架构，这才是这个工作的核心。其实DiT基本沿用了ViT的设计，如下图所示，首先采用一个**patch embedding来将输入进行patch化**，即得到一系列的tokens。其中patch size属于一个超参数，它直接决定了tokens的数量，这会影响模型的计算量。DiT的patch size共选择了三种设置：𝑝=2,4,8。注意token化之后，这里还要加上positional embeddings，这里采用非学习的sin-cosine位置编码。

![Untitled](Diffusion%20model%EF%BC%9ADiT,%20transformer%20only%208a22b30972f343d0befe0f0a373d8387/Untitled%202.png)

将输入token化之后，就可以像ViT那样接transformer blocks了。但是对于扩散模型来说，往往还需要在网络中嵌入额外的条件信息，这里的条件包括timesteps以及类别标签（如果是文生图就是文本，但是DiT这里并没有涉及）。要说明的一点是，无论是timesteps还是类别标签，都可以采用一个embedding来进行编码。DiT共设计了四种方案来实现两个额外embeddings的嵌入，具体如下：

1. **In-context conditioning**：将两个embeddings看成两个tokens合并在输入的tokens中，这种处理方式有点类似ViT中的cls token，实现起来比较简单，也不基本上不额外引入计算量。
2. **Cross-attention block**：将两个embeddings拼接成一个数量为2的序列，然后在transformer block中插入一个cross attention，条件embeddings作为cross attention的key和value；这种方式也是目前文生图模型所采用的方式，它需要额外引入15%的Gflops。
3. **Adaptive layer norm (adaLN) block**：采用adaLN，这里是将time embedding和class embedding相加，然后来回归scale和shift两个参数，这种方式也基本不增加计算量。
4. **adaLN-Zero block**：采用zero初始化的adaLN，这里是将adaLN的linear层参数初始化为zero，这样网络初始化时transformer block的残差模块就是一个identity函数；另外一点是，这里除了在LN之后回归scale和shift，还在每个残差模块结束之前回归一个scale，如上图所示。

论文对四种方案进行了对比试验，发现采用**adaLN-Zero**效果是最好的，所以DiT默认都采用这种方式来嵌入条件embeddings。

![Untitled](Diffusion%20model%EF%BC%9ADiT,%20transformer%20only%208a22b30972f343d0befe0f0a373d8387/Untitled%203.png)

这里也贴一下基于**adaLN-Zero**的DiT block的具体实现代码：

```python
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

        # zero init
        nn.init.constant_(adaLN_modulation[-1].weight, 0)
        nn.init.constant_(adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
```

虽然DiT发现**adaLN-Zero**效果是最好的，但是这种方式只适合这种只有类别信息的简单条件嵌入，因为只需要引入一个class embedding；但是对于文生图来说，其条件往往是序列的text embeddings，采用cross-attention方案可能是更合适的。 由于对输入进行了token化，所以在网络的最后还需要一个decoder来恢复输入的原始维度，DiT采用一个简单的linear层来实现，直接将每个token映射为𝑝×𝑝×2𝐶的tensor，然后再进行reshape来得到和原始输入空间维度一样的输出，但是特征维度大小是原来的2倍，分别用来预测噪音和方差。具体实现代码如下所示：

```python
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        
     nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
```

在具体性能上，最大的模型DiT-XL/2采用classifier free guidance可以在class-conditional image generation on ImageNet 256×256任务上实现当时的sota。

![Untitled](Diffusion%20model%EF%BC%9ADiT,%20transformer%20only%208a22b30972f343d0befe0f0a373d8387/Untitled%204.png)

虽然DiT看起来不错，但是只在ImageNet上生成做了实验，并没有扩展到大规模的文生图模型。而且在DiT之前，其实也有基于transformer架构的扩散模型研究工作，比如[U-ViT](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2209.12152)，目前也已经有将transformer应用在大规模文生图（基于扩散模型）的工作，比如[UniDiffuser](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2303.06555)，但是其实都没有受到太大的关注。目前主流的文生图模型还是采用基于UNet，**UNet本身也混合了卷积和attention，它的优势一方面是高效，另外一方面是不需要位置编码比较容易实现变尺度的生成**，这些对具体落地应用都是比较重要的。

### **参考**

- [Scalable Diffusion Models with Transformers](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2212.09748)
- [https://github.com/facebookresearch/DiT](https://link.zhihu.com/?target=https%3A//github.com/facebookresearch/DiT)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2112.10752)
- [Improved Denoising Diffusion Probabilistic Models](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2102.09672)