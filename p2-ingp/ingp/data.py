"""NeRF-synthetic (Blender) loader: images stay on the GPU as uint8, rays are generated on the fly."""
import json
import math
import os

import cv2
import numpy as np
import torch


class BlenderScene:
    def __init__(self, root, split, device="cuda", downscale=1):
        with open(os.path.join(root, f"transforms_{split}.json")) as f:
            meta = json.load(f)
        imgs, poses = [], []
        for fr in meta["frames"]:
            path = os.path.join(root, fr["file_path"].lstrip("./") + ".png")
            im = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGRA uint8
            if im is None:
                raise FileNotFoundError(path)
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
            if downscale > 1:
                im = cv2.resize(im, (im.shape[1] // downscale, im.shape[0] // downscale), interpolation=cv2.INTER_AREA)
            imgs.append(im)
            poses.append(np.array(fr["transform_matrix"], dtype=np.float32))
        self.images = torch.from_numpy(np.stack(imgs)).to(device)  # [N,H,W,4] uint8
        self.c2w = torch.from_numpy(np.stack(poses)).to(device)  # [N,4,4], OpenGL camera (looks down -z, y up)
        self.n, self.h, self.w = self.images.shape[:3]
        self.focal = 0.5 * self.w / math.tan(0.5 * meta["camera_angle_x"])
        self.device = device

    def rays(self, img_idx, px, py):
        """Ray origins/directions (unit length) for pixel centres (px, py) of images img_idx."""
        d = torch.stack([(px + 0.5 - 0.5 * self.w) / self.focal,
                         -(py + 0.5 - 0.5 * self.h) / self.focal,
                         -torch.ones_like(px, dtype=torch.float32)], -1)
        c2w = self.c2w[img_idx]
        d = (c2w[:, :3, :3] @ d[..., None])[..., 0]
        d = d / d.norm(dim=-1, keepdim=True)
        o = c2w[:, :3, 3]
        return o, d

    def pixels(self, img_idx, px, py):
        """RGBA in [0,1], float32."""
        return self.images[img_idx, py, px].float() / 255.0

    def sample_batch(self, n_rays, generator=None):
        idx = torch.randint(0, self.n, (n_rays,), device=self.device, generator=generator)
        px = torch.randint(0, self.w, (n_rays,), device=self.device, generator=generator)
        py = torch.randint(0, self.h, (n_rays,), device=self.device, generator=generator)
        o, d = self.rays(idx, px, py)
        return o, d, self.pixels(idx, px, py)

    def image_rays(self, i):
        py, px = torch.meshgrid(torch.arange(self.h, device=self.device), torch.arange(self.w, device=self.device), indexing="ij")
        px, py = px.reshape(-1), py.reshape(-1)
        idx = torch.full_like(px, i)
        o, d = self.rays(idx, px, py)
        return o, d, self.pixels(idx, px, py)
