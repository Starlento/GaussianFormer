import os
import numpy as np
import torch

from . import OPENOCC_TRANSFORMS


@OPENOCC_TRANSFORMS.register_module()
class LoadBEVLabelSurroundOcc(object):
    """Generate BEV labels from SurroundOcc occupancy annotations."""

    def __init__(self, occ_path, semantic=True, empty_label=17):
        self.occ_path = occ_path
        self.semantic = semantic
        self.empty_label = empty_label

        # Precompute meshgrid for occupancy xyz (same as SurroundOcc settings)
        self.xyz = self.get_meshgrid([-50, -50, -5.0, 50, 50, 3.0], [200, 200, 16], 0.5)
        # BEV coordinates taken from lowest height slice, z set to 0
        bev_xy = self.xyz[:, :, 0, :].copy()
        bev_xy[..., 2] = 0
        self.bev_xyz = bev_xy

    def get_meshgrid(self, ranges, grid, reso):
        xxx = torch.arange(grid[0], dtype=torch.float32) * reso + 0.5 * reso + ranges[0]
        yyy = torch.arange(grid[1], dtype=torch.float32) * reso + 0.5 * reso + ranges[1]
        zzz = torch.arange(grid[2], dtype=torch.float32) * reso + 0.5 * reso + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([xxx, yyy, zzz], dim=-1).numpy()
        return xyz  # shape (H, W, Z, 3)

    def __call__(self, results):
        label_file = os.path.join(self.occ_path, results['pts_filename'].split('/')[-1] + '.npy')
        if not os.path.exists(label_file):
            raise FileNotFoundError(f'{label_file} not found')

        label = np.load(label_file)
        occ_label = np.ones((200, 200, 16), dtype=np.int64) * self.empty_label
        occ_label[label[:, 0], label[:, 1], label[:, 2]] = label[:, 3]

        # Collapse along height to generate BEV label
        flat = occ_label.reshape(-1, 16)
        bev_label = np.ones(flat.shape[0], dtype=np.int64) * self.empty_label
        for idx in range(flat.shape[0]):
            col = flat[idx]
            col = col[col != self.empty_label]
            if col.size > 0:
                bev_label[idx] = np.bincount(col, minlength=self.empty_label + 1).argmax()
        bev_label = bev_label.reshape(200, 200)
        bev_mask = bev_label != self.empty_label

        results['bev_label'] = bev_label if self.semantic else bev_mask
        results['bev_cam_mask'] = bev_mask
        # full 3D coordinates for rendering
        results['occ_xyz'] = self.xyz.copy()
        # 2D BEV coordinates for loss supervision
        results['bev_xyz'] = self.bev_xyz.copy()
        return results

    def __repr__(self):
        return self.__class__.__name__
