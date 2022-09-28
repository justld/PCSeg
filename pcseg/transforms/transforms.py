import numpy as np
from typing import Tuple

import pcseg.transforms.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data


class LoadSemanticKITTI:
    def __init__(self,
                 project_label=True,
                 H=64,
                 W=1024,
                 fov_up=3.0,
                 fov_down=-25.0):
        self.project_label = project_label
        self.proj_H = H
        self.proj_W = W
        self.upper_inclination = fov_up / 180. * np.pi
        self.lower_inclination = -fov_down / 180. * np.pi
        self.fov = self.upper_inclination - self.lower_inclination

    def __call__(self, data):
        assert isinstance(
            data, dict
        ), "The data of LoadSemanticKITTI must be dict, but got {}.".format(
            type(data))
        assert 'image_path' in data.keys(
        ), "The data must include keys 'image_path'."
        raw_scan = np.fromfile(
            data['image_path'], dtype=np.float32).reshape((-1, 4))
        points = raw_scan[:, 0:3]
        remissions = raw_scan[:, 3]

        depth = np.linalg.norm(points, ord=2, axis=1)

        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(self.lower_inclination)
                        ) / self.fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        proj_x_copy = np.copy(
            proj_x
        )  # save a copy in original order, for each point, where it is in the range image

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        proj_y_copy = np.copy(
            proj_y
        )  # save a copy in original order, for each point, where it is in the range image

        # unproj_range_copy = np.copy(depth)   # copy of depth in original order

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = points[order]
        remission = remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # projected range image - [H,W] range (-1 is no data)
        proj_range = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        proj_remission = np.full(
            (self.proj_H, self.proj_W), -1, dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)

        proj_range[proj_y, proj_x] = depth
        proj_xyz[proj_y, proj_x] = points
        proj_remission[proj_y, proj_x] = remission
        proj_idx[proj_y, proj_x] = indices
        proj_mask = proj_idx > 0  # mask containing for each pixel, if it contains a point or not

        feat = np.concatenate([
            proj_range[None, ...], proj_xyz.transpose([2, 0, 1]),
            proj_remission[None, ...]
        ])

        if 'label_path' in data.keys():
            # load labels
            raw_label = np.fromfile(
                data['label_path'], dtype=np.uint32).reshape((-1))
            # only fill in attribute if the right size
            if raw_label.shape[0] != points.shape[0]:
                raise ValueError(
                    "Scan and Label don't contain same number of points. {}".
                    format(data['label_path']))

            if self.project_label:
                proj_sem_label = np.zeros(
                    (self.proj_H, self.proj_W), dtype=np.int32)  # [H,W]  label
                proj_sem_label[proj_mask] = raw_label[proj_idx[proj_mask]]

                label = proj_sem_label.astype(np.int64)[None, ...]
            else:
                label = raw_label.astype(np.int64)
        data = {
            'data': feat,
            'proj_mask': proj_mask.astype(np.float32),
            'proj_x': proj_x_copy,
            'proj_y': proj_y_copy,
            'label': label,
        }
        return data


class NormalizeRangeImage:
    def __init__(self,
                 mean: Tuple[float, float, float],
                 std: Tuple[float, float, float]):
        if not (isinstance(mean, (list, tuple)) and isinstance(std,
                                                               (list, tuple))):
            raise ValueError(
                "{}: input type is invalid. It should be list or tuple".format(
                    self))

        from functools import reduce
        if reduce(lambda x, y: x * y, std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

        self.mean = np.array(mean)[:, None, None]
        self.std = np.array(std)[:, None, None]

    def __call__(self, data):
        data['data'] = F.normalize(data['data'], self.mean, self.std)
        return data
