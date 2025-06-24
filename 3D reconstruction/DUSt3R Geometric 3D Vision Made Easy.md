# DUSt3R: Geometric 3D Vision Made Easy

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image.png)

# Introduction

**Dust3R** addresses **dense and unconstrained stereo 3D reconstruction** from arbitrary image collections—operating **without** any prior knowledge of **camera intrinsics or extrinsics**. It introduces a **simple yet effective global alignment strategy** that registers multiple views by expressing all pairwise **pointmaps** in a **shared reference frame**.

The core architecture is a **Transformer-based encoder-decoder**, built on **pre-trained models**, that jointly processes the **scene** and the **input images**. Given a set of **unconstrained images** as input, Dust3R outputs **dense depth maps**, **pixel-wise correspondences**, and both **relative** and **absolute camera poses**, enabling the reconstruction of a consistent **3D model**.

Dust3R is trained in a **fully-supervised** manner using a simple regression loss. It leverages large-scale datasets with **ground-truth annotations**, either synthetically generated or derived from SfM pipelines. Notably, during inference, **no geometric constraints or priors** are required—making it flexible and broadly applicable.

***We directly regress dense 3D point maps for each image, warped into a common coordinate frame (typically that of view1), which enables us to get pixel-aligned correspondences without matching.***

# Method

### Pointmap

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%201.png)

### Cameras and scene

Note: Depth image is represented with depth (not normalized invert depth)

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%202.png)

## Overview

### Inputs and outputs

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%203.png)

### **Network architecture**

Two **identical** branch：an encoder, a decoder + a head

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%204.png)

- **Siamese encoding** enables consistent feature extraction. （ViT）

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%205.png)

- **Generic transformer decorder**: self attention(with token+img token)+ **cross attention** + MLP
- **Cross-attention** allows views to communicate, aligning their 3D predictions.
- **Pointmaps** are generated per view but aligned through decoder interaction. (aligned to camera 1 coordinate)

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%206.png)

### **Pointmap Output and Architectural Design**

- The predicted **pointmaps** are only accurate **up to an unknown scale**.
- The architecture **does not enforce any explicit geometric constraints** (e.g., no pinhole camera model), so the pointmaps may **not strictly follow real-world camera geometry**.
- The model **learns geometrically consistent pointmaps**.
- This architecture enables the use of **powerful pretrained models**

## Training Objective

**3D Regression loss.**

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%207.png)

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%208.png)

**Confidence-aware loss.**

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%209.png)

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%2010.png)

## Downstream Applications

### Point matching

nearest neighbor (NN) search in the 3D pointmap space. Reciprocal (**Mutual)** correspondences.

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%2011.png)

### Recovering intrinsics

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%2012.png)

### Relative pose estimation

1. 2D match + camera intrinsic + essential matrix estimation
2. Procrustes alignment (close form, sensitive to noise and outliers, inference twice)
    
    ![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%2013.png)
    
3. **Ransac+PNP**

### Absolute pose estimation

get the relative pose between IQ and IB as described previously. Then, we convert this pose to **world coordinate by scaling** it appropriately, according to the scale between XB,B and the ground-truth pointmap for IB.

## Global alignment

post processing to a joint 3D space with **aligned 3D point-clouds** and their corresponding **pixel-to-3D mapping**

 **pairwise graph:** all pairs ⇒ image retrival method (AP-GeM [95]), **filter low average confidence** pairs. (nodes: images, edge：images share visual content)

**Global optimization**：

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%2014.png)

### Recovering camera parameters

enforcing a standard camera pinhole model

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%2015.png)

*why not BA？ too long to do optimization* 

this method took mere seconds on standard GPU

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%2016.png)

# Experiments

### Find image pairs

![image.png](images/DUSt3R%20Geometric%203D%20Vision%20Made%20Easy%201ba71bdab3cf80a08e7afbedcb4a1605/image%2017.png)

### Train details

1. first, 224*224, then larger 512 images. randomly select image aspect ratios (crop→ largest dim to 512)
2. standard image augmentation
3. Network: **Vit-large for the encoder, Vit-base the decoder and a DPT head**.

```python
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
import torch
import os
from packaging import version
import huggingface_hub

from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), ("Outdated huggingface_hub version, "
                                                                     "please reinstall requirements.txt")

def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)

class AsymmetricCroCo3DStereo (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            try:
                model = super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)
            except TypeError as e:
                raise Exception(f'tried to load {pretrained_model_name_or_path} from huggingface, but failed')
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none': [],
            'mask': [self.mask_token],
            'encoder': [self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2

```

Loss function：convert GT and predictions to view1 for loss computation

```python
def get_pred_pts3d(gt, pred, use_pose=False):
    if 'depth' in pred and 'pseudo_focal' in pred:
        try:
            pp = gt['camera_intrinsics'][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif 'pts3d' in pred:
        # pts3d from my camera
        pts3d = pred['pts3d']

    elif 'pts3d_in_other_view' in pred:
        # pts3d from the other camera, already transformed
        assert use_pose is True
        return pred['pts3d_in_other_view']  # return!

    if use_pose:
        camera_pose = pred.get('camera_pose')
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d
    
class Regr3D (Criterion, MultiLoss):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
              = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.gt_scale = gt_scale

    def get_all_pts3d(self, gt1, gt2, pred1, pred2, dist_clip=None):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gt1['camera_pose'])
        gt_pts1 = geotrf(in_camera1, gt1['pts3d'])  # B,H,W,3
        gt_pts2 = geotrf(in_camera1, gt2['pts3d'])  # B,H,W,3

        valid1 = gt1['valid_mask'].clone()
        valid2 = gt2['valid_mask'].clone()

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis1 = gt_pts1.norm(dim=-1)  # (B, H, W)
            dis2 = gt_pts2.norm(dim=-1)  # (B, H, W)
            valid1 = valid1 & (dis1 <= dist_clip)
            valid2 = valid2 & (dis2 <= dist_clip)

        pr_pts1 = get_pred_pts3d(gt1, pred1, use_pose=False)
        pr_pts2 = get_pred_pts3d(gt2, pred2, use_pose=True)

        # normalize 3d points
        if self.norm_mode:
            pr_pts1, pr_pts2 = normalize_pointcloud(pr_pts1, pr_pts2, self.norm_mode, valid1, valid2)
        if self.norm_mode and not self.gt_scale:
            gt_pts1, gt_pts2 = normalize_pointcloud(gt_pts1, gt_pts2, self.norm_mode, valid1, valid2)

        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, {}

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, monitoring = \
            self.get_all_pts3d(gt1, gt2, pred1, pred2, **kw)
        # loss on img1 side
        l1 = self.criterion(pred_pts1[mask1], gt_pts1[mask1])
        # loss on gt2 side
        l2 = self.criterion(pred_pts2[mask2], gt_pts2[mask2])
        self_name = type(self).__name__
        details = {self_name + '_pts3d_1': float(l1.mean()), self_name + '_pts3d_2': float(l2.mean())}
        return Sum((l1, mask1), (l2, mask2)), (details | monitoring)

class ConfLoss (MultiLoss):
    """ Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10) 

        alpha: hyperparameter
    """

    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction('none')

    def get_name(self):
        return f'ConfLoss({self.pixel_loss})'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        # compute per-pixel loss
        ((loss1, msk1), (loss2, msk2)), details = self.pixel_loss(gt1, gt2, pred1, pred2, **kw)
        if loss1.numel() == 0:
            print('NO VALID POINTS in img1', force=True)
        if loss2.numel() == 0:
            print('NO VALID POINTS in img2', force=True)

        # weight by confidence
        conf1, log_conf1 = self.get_conf_log(pred1['conf'][msk1])
        conf2, log_conf2 = self.get_conf_log(pred2['conf'][msk2])
        conf_loss1 = loss1 * conf1 - self.alpha * log_conf1
        conf_loss2 = loss2 * conf2 - self.alpha * log_conf2

        # average + nan protection (in case of no valid pixels at all)
        conf_loss1 = conf_loss1.mean() if conf_loss1.numel() > 0 else 0
        conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0

        return conf_loss1 + conf_loss2, dict(conf_loss_1=float(conf_loss1), conf_loss2=float(conf_loss2), **details)
```
