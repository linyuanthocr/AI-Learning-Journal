# DINO v2

DINOv2: Learning Robust Visual Features without Supervision

http://github.com/facebookresearch/dinov2

https://arxiv.org/abs/2304.07193

https://dinov2.metademolab.com/

### Contributions

1. self-supervised learning can produce general purpose visual features features that work across image distributions and tasks without finetuning
2. data pipeline: automatic pipeline to filter and rebalance datasets from an extensive collection of uncurated images（based on data similiarity， 142M images）
3. ViT(1B parameters) + distill into a series of small models.
4. stabilizing and accelerating discriminative self-supervised learning when scaling in model and data sizes. (2× faster and require 3× less memory)

### Data processing

![image.png](images/DINO%20v2%20a6b36d0fde2a4a929940fd12bfd21c5b/image.png)

1. data source: curated + uncurated(1.2B)
    1. uncurated data ⇒ domain selection + post processing(PCA hash deduplication, NSFW filtering， blurring id)
2. Deduplication
    1. For train and test set. (SSCD, self-supervised copy detection)
    2. SSCD: [SimCLR](https://www.notion.so/SimCLR-1ba71bdab3cf8046b334f4fafe0ca8af?pvs=21), InfoNCE
    
    ![image.png](images/DINO%20v2%20a6b36d0fde2a4a929940fd12bfd21c5b/image%201.png)
    
    ![image.png](images/DINO%20v2%20a6b36d0fde2a4a929940fd12bfd21c5b/image%202.png)
    
    ![image.png](images/DINO%20v2%20a6b36d0fde2a4a929940fd12bfd21c5b/image%203.png)
    

1. Self-supervised image retrieval
    
    Embedding (encoder: Vit-H/16 pretrained on imageNet22k) + KNN (N=4)
    

### Discriminative Self-supervised Pre-training

1. **Image level objective** 
We consider the **cross-entropy loss** between the features extracted from **a student and a teacher network**. Both features are coming from **the class token of a ViT**, obtained from different crops of the same image. We pass the student class token through the student DINO head. This **head is an MLP mode**l outputting a vector of scores, that we call "prototype scores". We then apply a **softmax** to obtain ps. Similarly, we apply the teacher DINO head to the teacher class token to obtain teacher prototype scores. We then apply a softmax followed by a **centering with moving average** (or a Sinkhorn-Knopp centering as detailed thereafter) to obtain pt. The DINO loss term corresponds to:
    
    ![image.png](images/DINO%20v2%20a6b36d0fde2a4a929940fd12bfd21c5b/image%204.png)
    
    *We learn the parameters of the student and build the teacher head with an exponential moving average of past iterates (He et al., 2020)*
    
2. **Patch level objective**
    
    We **randomly mask some of the input patches given to the student**, but not to the teacher. We then apply the student **iBOT head** to the student mask tokens. Similarly, we apply the teacher iBOT head to the (visible) teacher patch tokens corresponding to the ones masked in the student. We then apply the **softmax and centering** steps as above, and obtain the iBOT loss term
    
    ![image.png](images/DINO%20v2%20a6b36d0fde2a4a929940fd12bfd21c5b/image%205.png)
    
    *We learn the parameters of the student and build the teacher head with an exponential moving average of past iterates (He et al., 2020)*
    
3. **Untying head weights between both objectives.** 
    
    two separate heads for iBOT and DINO
    
4. **Sinkhorn-Knopp centering (for teacher)**
5. **KoLeo regularizer**
    
    ![image.png](images/DINO%20v2%20a6b36d0fde2a4a929940fd12bfd21c5b/image%206.png)
    
6. Adapting the resolution
    
    increase the resolution of images to **518×518** during a short period **at the end of pretraining**
    

### Efficient implementation

1. FlashAttention (dim is a multple of 256)
2. Sequence packing: single long sequence+block-diagonal mask in self-attention.
3. Efficient stochastic depth: skips the computation of the dropped residuals rather than masking the result.
4. Fully-Sharded Data Parallel (FSDP). 
    1. 4 model replicas in float32 precision – student, teacher, optimizer first moments, optimizer second moments.
    2. the weight shards are stored in float32 precision as required by the optimizer, but broadcasting weights and reducing gradients is done in float16 precision for the backbone
5. Model distillation
    1. smaller model: distill them from largest model. 
    2. we use a larger model as a frozen teacher, keep a spare EMA of the student that we use as our final model, remove the masking and stochastic depth, and, apply the iBOT loss on the two global crops. In

### **Depth Estimation**

We consider three different setups for this evaluation. lin. 1: we extract the last layer of the frozen transformer and concatenate **the [CLS] token to each patch token**. Then we bi-linearly **upsample** the tokens by a factor of 4 to increase the resolution. Finally we train a simple linear layer using a classification loss by dividing the depth prediction range in 256 uniformly distributed bins and use a linear normalization following Bhat et al. (2021). lin. 4: we use the same protocol that we use with one layer, but concatenate the tokens from layers l = {3, 6, 9, 12} for ViT-S/B, l = {5, 12, 18, 24} for ViT-L, and l = {10, 20, 30, 40} for ViT-g. **DPT**: we use the **DPT** **decoder** (Ranftl et al., 2021) on top of **our frozen models** and setup a regression task. We scale the size of the head following the dimension of the features for each architecture. We show results for all baselines, all datasets and all setups in Table 11.

![image.png](images/DINO%20v2%20a6b36d0fde2a4a929940fd12bfd21c5b/image%207.png)
