# Aggregated Selective Match Kernels (ASMK)

https://arxiv.org/abs/1812.01584

![image.png](images/Aggregated%20Selective%20Match%20Kernels%20ASMK%2021c71bdab3cf80dfa08ed81f2119a8d8/image.png)

ASMK (Aggregated Selective Match Kernel) is an advanced image retrieval algorithm that combines the best of two worlds:

1. **Selective Matching:** It focuses on high-confidence local feature matches between images, similar to traditional feature matching methods. This helps to filter out noisy or irrelevant matches.
2. **Feature Aggregation:** It aggregates a large number of local features into a compact, global image representation (like VLAD or Fisher Vector). This makes the representation robust to "burstiness" (when many similar features appear in one area) and efficient for large-scale databases.

**In essence, ASMK works by:**

- **Extracting and quantifying local features** (e.g., SIFT descriptors).
- **Calculating "residual vectors"** for each feature, which capture the difference between the feature and its assigned visual word.
- Applying a **"selective match kernel"** to ensure only strong, meaningful matches contribute to the similarity score.
- **Aggregating these selected residual vectors** into a concise image signature.
- Optionally, **compressing and binarizing** this signature (ASMK*) for even greater efficiency in large-scale scenarios.

**The benefits of ASMK are:**

- **High performance** in image retrieval.
- **Robustness** to common image challenges like "burstiness."
- **Scalability** for large databases due to efficient representation and optional compression.

ASMK is widely used in content-based image retrieval and other computer vision tasks requiring robust image similarity comparisons.

### **🔄 ASMK Pipeline (as used in the code)**

Here’s a high-level breakdown of the pipeline:

1. **Local Feature Extraction**
    
    Using a retrieval backbone (e.g. MASt3R), it extracts dense local features per image:
    

```
feat, ids = extract_local_features(self.model, impaths, self.imsize)
```

1. **ASMK Codebook Training / Loading**

```
self.asmk = self.asmk.train_codebook(None, cache_path=cache_codebook_fname)
```

The codebook is usually loaded from a .pkl file trained previously. It’s a visual vocabulary built using clustering (e.g. k-means on local descriptors).

**Each image has finally become a D-dim** feature, with sum(feature*attention), **weighted** **SPoC**  (Sum-pooled convolutional features )

![image.png](images/Aggregated%20Selective%20Match%20Kernels%20ASMK%2021c71bdab3cf80dfa08ed81f2119a8d8/image%201.png)

1. **Build Inverted File Index (IVF)**

```
asmk_dataset = self.asmk.build_ivf(feat, ids)
```

Local features are quantized into visual words. Residuals (differences between descriptors and centroids) are stored in the inverted index.

kmeans clustering based ivf. 

1. **Querying**

```
metadata, query_ids, ranks, ranked_scores = asmk_dataset.query_ivf(feat, ids)
```

The same image set is queried. For each query image:

- Extract its residuals.
- Match them against indexed ones using **ASMK similarity**.
- Return ranks and scores.
1. **Scoring Matrix Output**
    
    The final pairwise image similarity matrix is:
    

```
scores = np.empty_like(ranked_scores)
scores[np.arange(ranked_scores.shape[0])[:, None], ranks] = ranked_scores
```

---

### **🧠 What Makes ASMK Special?**

- It uses **visual words + residuals** to encode local feature distributions per image.
- Selective: only top-ranked residuals contribute.
- Aggregated: aggregates residual vectors into a compact global descriptor.

```python
class Retriever(object):
    def __init__(self, modelname, backbone=None, device='cuda'):
        # load the model
        assert os.path.isfile(modelname), modelname
        print(f'Loading retrieval model from {modelname}')
        ckpt = torch.load(modelname, 'cpu')  # TODO from pretrained to download it automatically
        ckpt_args = ckpt['args']
        if backbone is None:
            backbone = AsymmetricMASt3R.from_pretrained(ckpt_args.pretrained)
        self.model = RetrievalModel(
            backbone, freeze_backbone=ckpt_args.freeze_backbone, prewhiten=ckpt_args.prewhiten,
            hdims=list(map(int, ckpt_args.hdims.split('_'))) if len(ckpt_args.hdims) > 0 else "",
            residual=getattr(ckpt_args, 'residual', False), postwhiten=ckpt_args.postwhiten,
            featweights=ckpt_args.featweights, nfeat=ckpt_args.nfeat
        ).to(device)
        self.device = device
        msg = self.model.load_state_dict(ckpt['model'], strict=False)
        assert all(k.startswith('backbone') for k in msg.missing_keys)
        assert len(msg.unexpected_keys) == 0
        self.imsize = ckpt_args.imsize

        # load the asmk codebook
        dname, bname = os.path.split(modelname)  # TODO they should both be in the same file ?
        bname_splits = bname.split('_')
        cache_codebook_fname = os.path.join(dname, '_'.join(bname_splits[:-1]) + '_codebook.pkl')
        assert os.path.isfile(cache_codebook_fname), cache_codebook_fname
        asmk_params = {'index': {'gpu_id': 0}, 'train_codebook': {'codebook': {'size': '64k'}},
                       'build_ivf': {'kernel': {'binary': True}, 'ivf': {'use_idf': False},
                                     'quantize': {'multiple_assignment': 1}, 'aggregate': {}},
                       'query_ivf': {'quantize': {'multiple_assignment': 5}, 'aggregate': {},
                                     'search': {'topk': None},
                                     'similarity': {'similarity_threshold': 0.0, 'alpha': 3.0}}}
        asmk_params['train_codebook']['codebook']['size'] = ckpt_args.nclusters
        self.asmk = asmk_method.ASMKMethod.initialize_untrained(asmk_params)
        self.asmk = self.asmk.train_codebook(None, cache_path=cache_codebook_fname)

    def __call__(self, input_imdir_or_imlistfile, outfile=None):
        # get impaths
        if isinstance(input_imdir_or_imlistfile, str):
            impaths = get_impaths_from_imdir_or_imlistfile(input_imdir_or_imlistfile)
        else:
            impaths = input_imdir_or_imlistfile  # we're assuming a list has been passed
        print(f'Found {len(impaths)} images')

        # build the database
        feat, ids = extract_local_features(self.model, impaths, self.imsize, tocpu=True, device=self.device)
        feat = feat.cpu().numpy()
        ids = ids.cpu().numpy()
        asmk_dataset = self.asmk.build_ivf(feat, ids)

        # we actually retrieve the same set of images
        metadata, query_ids, ranks, ranked_scores = asmk_dataset.query_ivf(feat, ids)

        # well ... scores are actually reordered according to ranks ...
        # so we redo it the other way around...
        scores = np.empty_like(ranked_scores)
        scores[np.arange(ranked_scores.shape[0])[:, None], ranks] = ranked_scores

        # save
        if outfile is not None:
            if os.path.isdir(os.path.dirname(outfile)):
                os.makedirs(os.path.dirname(outfile), exist_ok=True)
            np.save(outfile, scores)
            print(f'Scores matrix saved in {outfile}')
        return scores

```

Whitening:

![image.png](images/Aggregated%20Selective%20Match%20Kernels%20ASMK%2021c71bdab3cf80dfa08ed81f2119a8d8/image%202.png)

```
# from https://github.com/gtolias/how/blob/4d73c88e0ffb55506e2ce6249e2a015ef6ccf79f/how/utils/whitening.py#L20
def pcawhitenlearn_shrinkage(X, s=1.0):
    """Learn PCA whitening with shrinkage from given descriptors"""
    N = X.shape[0]

    # Learning PCA w/o annotations
    m = X.mean(axis=0, keepdims=True)
    Xc = X - m
    Xcov = np.dot(Xc.T, Xc)
    Xcov = (Xcov + Xcov.T) / (2 * N)
    eigval, eigvec = np.linalg.eig(Xcov)
    order = eigval.argsort()[::-1]
    eigval = eigval[order]
    eigvec = eigvec[:, order]

    eigval = np.clip(eigval, a_min=1e-14, a_max=None)
    P = np.dot(np.linalg.inv(np.diag(np.power(eigval, 0.5 * s))), eigvec.T)

    return m, P.T
```

```python
def weighted_spoc(feat, attn):
    """
    feat: BxNxC
    attn: BxN
    output: BxC L2-normalization weighted-sum-pooling of features
    """
    return torch.nn.functional.normalize((feat * attn[:, :, None]).sum(dim=1), dim=1)

def how_select_local(feat, attn, nfeat):
    """
    feat: BxNxC
    attn: BxN
    nfeat: nfeat to keep
    """
    # get nfeat
    if nfeat < 0:
        assert nfeat >= -1.0
        nfeat = int(-nfeat * feat.size(1))
    else:
        nfeat = int(nfeat)
    # asort
    topk_attn, topk_indices = torch.topk(attn, min(nfeat, attn.size(1)), dim=1)
    topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, feat.size(2))
    topk_features = torch.gather(feat, 1, topk_indices_expanded)
    return topk_features, topk_attn, topk_indices

class RetrievalModel(nn.Module):
    def __init__(self, backbone, freeze_backbone=1, prewhiten=None, hdims=[1024], residual=False, postwhiten=None,
                 featweights='l2norm', nfeat=300, pretrained_retrieval=None):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.backbone_dim = backbone.enc_embed_dim
        self.prewhiten = nn.Identity() if prewhiten is None else Whitener(self.backbone_dim)
        self.prewhiten_freq = prewhiten
        if prewhiten is not None and prewhiten != -1:
            for p in self.prewhiten.parameters():
                p.requires_grad = False
        self.residual = residual
        self.projector = self.build_projector(hdims, residual)
        self.dim = hdims[-1] if len(hdims) > 0 else self.backbone_dim
        self.postwhiten_freq = postwhiten
        self.postwhiten = nn.Identity() if postwhiten is None else Whitener(self.dim)
        if postwhiten is not None and postwhiten != -1:
            assert len(hdims) > 0
            for p in self.postwhiten.parameters():
                p.requires_grad = False
        self.featweights = featweights
        if featweights == 'l2norm':
            self.attention = lambda x: x.norm(dim=-1)
        else:
            raise NotImplementedError(featweights)
        self.nfeat = nfeat
        self.pretrained_retrieval = pretrained_retrieval
        if self.pretrained_retrieval is not None:
            ckpt = torch.load(pretrained_retrieval, 'cpu')
            msg = self.load_state_dict(ckpt['model'], strict=False)
            assert len(msg.unexpected_keys) == 0 and all(k.startswith('backbone')
                                                         or k.startswith('postwhiten') for k in msg.missing_keys)

    def build_projector(self, hdims, residual):
        if self.residual:
            assert hdims[-1] == self.backbone_dim
        d = self.backbone_dim
        if len(hdims) == 0:
            return nn.Identity()
        layers = []
        for i in range(len(hdims) - 1):
            layers.append(nn.Linear(d, hdims[i]))
            d = hdims[i]
            layers.append(nn.LayerNorm(d))
            layers.append(nn.GELU())
        layers.append(nn.Linear(d, hdims[-1]))
        return nn.Sequential(*layers)

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        ss = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        if self.freeze_backbone:
            ss = {k: v for k, v in ss.items() if not k.startswith('backbone')}
        return ss

    def reinitialize_whitening(self, epoch, train_dataset, nimgs=5000, log_writer=None, max_nfeat_per_image=None, seed=0, device=default_device):
        do_prewhiten = self.prewhiten_freq is not None and self.pretrained_retrieval is None and \
            (epoch == 0 or (self.prewhiten_freq > 0 and epoch % self.prewhiten_freq == 0))
        do_postwhiten = self.postwhiten_freq is not None and ((epoch == 0 and self.postwhiten_freq in [0, -1])
                                                              or (self.postwhiten_freq > 0 and
                                                                  epoch % self.postwhiten_freq == 0 and epoch > 0))
        if do_prewhiten or do_postwhiten:
            self.eval()
            imdataset = train_dataset.imlist_dataset_n_images(nimgs, seed)
            loader = torch.utils.data.DataLoader(imdataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        if do_prewhiten:
            print('Re-initialization of pre-whitening')
            t = time.time()
            with torch.no_grad():
                features = []
                for d in tqdm(loader):
                    feat = self.backbone._encode_image(d['img'][0, ...].to(device),
                                                       true_shape=d['true_shape'][0, ...])[0]
                    feat = feat.flatten(0, 1)
                    if max_nfeat_per_image is not None and max_nfeat_per_image < feat.size(0):
                        l2norms = torch.linalg.vector_norm(feat, dim=1)
                        feat = feat[torch.argsort(-l2norms)[:max_nfeat_per_image], :]
                    features.append(feat.cpu())
            features = torch.cat(features, dim=0)
            features = features.numpy()
            m, P = pcawhitenlearn_shrinkage(features)
            self.prewhiten.load_state_dict({'m': torch.from_numpy(m), 'p': torch.from_numpy(P)})
            prewhiten_time = time.time() - t
            print(f'Done in {prewhiten_time:.1f} seconds')
            if log_writer is not None:
                log_writer.add_scalar('time/prewhiten', prewhiten_time, epoch)
        if do_postwhiten:
            print(f'Re-initialization of post-whitening')
            t = time.time()
            with torch.no_grad():
                features = []
                for d in tqdm(loader):
                    backbone_feat = self.backbone._encode_image(d['img'][0, ...].to(device),
                                                                true_shape=d['true_shape'][0, ...])[0]
                    backbone_feat_prewhitened = self.prewhiten(backbone_feat)
                    proj_feat = self.projector(backbone_feat_prewhitened) + \
                        (0.0 if not self.residual else backbone_feat_prewhitened)
                    proj_feat = proj_feat.flatten(0, 1)
                    if max_nfeat_per_image is not None and max_nfeat_per_image < proj_feat.size(0):
                        l2norms = torch.linalg.vector_norm(proj_feat, dim=1)
                        proj_feat = proj_feat[torch.argsort(-l2norms)[:max_nfeat_per_image], :]
                    features.append(proj_feat.cpu())
                features = torch.cat(features, dim=0)
                features = features.numpy()
            m, P = pcawhitenlearn_shrinkage(features)
            self.postwhiten.load_state_dict({'m': torch.from_numpy(m), 'p': torch.from_numpy(P)})
            postwhiten_time = time.time() - t
            print(f'Done in {postwhiten_time:.1f} seconds')
            if log_writer is not None:
                log_writer.add_scalar('time/postwhiten', postwhiten_time, epoch)

    def extract_features_and_attention(self, x):
        backbone_feat = self.backbone._encode_image(x['img'], true_shape=x['true_shape'])[0]
        backbone_feat_prewhitened = self.prewhiten(backbone_feat)
        proj_feat = self.projector(backbone_feat_prewhitened) + \
            (0.0 if not self.residual else backbone_feat_prewhitened)
        attention = self.attention(proj_feat)
        proj_feat_whitened = self.postwhiten(proj_feat)
        return proj_feat_whitened, attention

    def forward_local(self, x):
        feat, attn = self.extract_features_and_attention(x)
        return how_select_local(feat, attn, self.nfeat)

    def forward_global(self, x):
        feat, attn = self.extract_features_and_attention(x)
        return weighted_spoc(feat, attn)

    def forward(self, x):
        return self.forward_global(x)
```

```python
@torch.no_grad()
def extract_local_features(model, images, imsize, seed=0, tocpu=False, max_nfeat_per_image=None,
                           max_nfeat_per_image2=None, device=default_device):
    model.eval()
    imdataset = Dust3rInputFromImageList(images, imsize=imsize) if isinstance(images, list) else images
    loader = torch.utils.data.DataLoader(imdataset, batch_size=1, shuffle=False,
                                         num_workers=8, pin_memory=True, collate_fn=identity)
    with torch.no_grad():
        features = []
        imids = []
        for i, d in enumerate(tqdm(loader)):
            dd = d[0]
            dd['img'] = dd['img'].to(device, non_blocking=True)
            feat, _, _ = model.forward_local(dd)
            feat = feat.flatten(0, 1)
            if max_nfeat_per_image is not None and feat.size(0) > max_nfeat_per_image:
                feat = feat[torch.randperm(feat.size(0))[:max_nfeat_per_image], :]
            if max_nfeat_per_image2 is not None and feat.size(0) > max_nfeat_per_image2:
                feat = feat[:max_nfeat_per_image2, :]
            features.append(feat)
            if tocpu:
                features[-1] = features[-1].cpu()
            imids.append(i * torch.ones_like(features[-1][:, 0]).to(dtype=torch.int64))
    features = torch.cat(features, dim=0)
    imids = torch.cat(imids, dim=0)
    return features, imids
```
