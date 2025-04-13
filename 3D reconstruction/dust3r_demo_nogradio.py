import argparse
import os
import copy
from pathlib import Path
import torch
import matplotlib.pyplot as pl

from dust3r.utils.image import load_images, rgb
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from scipy.spatial.transform import Rotation
import numpy as np
import trimesh

def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = Path(outdir) / 'scene.glb'
    outfile.parent.mkdir(parents=True, exist_ok=True)
    if not silent:
        print(f'(exporting 3D scene to {outfile})')
    scene.export(file_obj=str(outfile))
    return str(outfile)

def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05):
    if scene is None:
        return None
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world,
                                        as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)

def get_reconstructed_scene(outdir, model, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, refid):
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=not silent)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size)

    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths = [d / depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d / confs_max) for d in confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))

    return scene, outfile, imgs

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AsymmetricCroCo3DStereo.from_pretrained(args.model_name).to(device).eval()

    filelist = sorted([str(Path(args.image_path) / f) for f in os.listdir(args.image_path)
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    _, outfile, _ = get_reconstructed_scene(
        outdir=args.output_dir,
        model=model,
        device=device,
        silent=args.silent,
        image_size=args.image_size,
        filelist=filelist,
        schedule=args.schedule,
        niter=args.niter,
        min_conf_thr=args.min_conf_thr,
        as_pointcloud=args.as_pointcloud,
        mask_sky=args.mask_sky,
        clean_depth=args.clean_depth,
        transparent_cams=args.transparent_cams,
        cam_size=args.cam_size,
        scenegraph_type="complete",
        winsize=1,
        refid=0)
    print("✅ Done. Output:", outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
                        help="Path to model weights")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to folder of input images")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Resize longest side to this value")
    parser.add_argument("--schedule", type=str, default="cosine")
    parser.add_argument("--niter", type=int, default=300)
    parser.add_argument("--min_conf_thr", type=float, default=3.0)
    parser.add_argument("--as_pointcloud", action="store_true")
    parser.add_argument("--mask_sky", action="store_true")
    parser.add_argument("--clean_depth", action="store_true")
    parser.add_argument("--transparent_cams", action="store_true")
    parser.add_argument("--cam_size", type=float, default=0.05)
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()
    main(args)

