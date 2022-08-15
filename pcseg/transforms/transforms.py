# Copyright (c) 2022 justld Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import numpy as np

from pcseg.cvlibs import manager


@manager.TRANSFORMS.add_component
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs):
        for trans in self.transforms:
            inputs = trans(inputs)
        return inputs


@manager.TRANSFORMS.add_component
class RandomFeatContrast:
    def __init__(self, prob=0.2, blend_factor=None):
        self.prob = prob
        self.blend_factor = blend_factor

    def __call__(self, data):
        if np.random.rand() < self.prob:
            lo = np.min(data['feat'][:, :3], 0, keepdims=True)
            hi = np.max(data['feat'][:, :3], 0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_feat = (data['feat'][:, :3] - lo) * scale
            blend_factor = np.random.rand(
            ) if self.blend_factor is None else self.blend_factor
            data['feat'][:, :3] = (
                1 - blend_factor
            ) * data['feat'][:, :3] + blend_factor * contrast_feat
        return data


@manager.TRANSFORMS.add_component
class RandomPositionScaling:
    """
    Random scaling position in point cloud data.

    Args:
        scale_range (list, optional): The random scaling ratio range. Default: [2. / 3, 3. / 2].
        anisotropic (bool, optional): Whether you use same scale ratio in different axis. Default: True.
        scale_xyz (list, optional):  Whether use scaling in xyz axis. Default: [True, True, True].
    """

    def __init__(self,
                 scale_range=[2. / 3, 3. / 2],
                 anisotropic=True,
                 scale_xyz=[True, True, True]):
        self.scale_min, self.scale_max = np.array(scale_range).astype(
            np.float32)
        self.anisotropic = anisotropic
        self.scale_xyz = scale_xyz

    def __call__(self, data):
        scale = np.random.rand(3 if self.anisotropic else 1) * (
            self.scale_max - self.scale_min) + self.scale_min
        for i, s in enumerate(self.scale_xyz):
            if not s:
                scale[i] = 1
        data['pos'] *= scale
        return data


@manager.TRANSFORMS.add_component
class PositionFloorCentering:
    """
    Centering the point cloud in the xy plane

    Args:
        gravity_dim (int): The dimension of gravity. Default: 2.
    """

    def __init__(self, gravity_dim=2):
        self.gravity_dim = gravity_dim

    def __call__(self, data):
        data['pos'] -= np.mean(data['pos'], axis=0, keepdims=True)
        data['pos'][:, self.gravity_dim] -= np.min(data['pos']
                                                   [:, self.gravity_dim])
        return data


@manager.TRANSFORMS.add_component
class PositionJitter(object):
    """
    Position noise.

    Args:
        jitter_sigma (float, optional): Noise scale ratio. Default: 0.01.
        jitter_clip (float, optional): Clipped noise range [-jitter_clip, jitter_clip]. Default: 0.05.
    """

    def __init__(self, jitter_sigma=0.01, jitter_clip=0.05):
        self.noise_std = jitter_sigma
        self.noise_clip = jitter_clip

    def __call__(self, data):
        noise = np.random.randn(*data['pos'].shape) * self.noise_std
        data['pos'] += np.clip(noise, -self.noise_clip, self.noise_clip)
        return data


@manager.TRANSFORMS.add_component
class RandomFeatDrop:
    """
    Random drop feat data.

    Args:
        color_drop (float, optional): Feature drop ratio. Default: 0.2.
    """

    def __init__(self, color_drop=0.2):
        self.color_drop = color_drop

    def __call__(self, data):
        colors_drop = random.random() < self.color_drop
        if colors_drop:
            data['feat'][:, :3] = 0
        return data


@manager.TRANSFORMS.add_component
class FeatNormalize:
    def __init__(self, color_mean=[0.5, 0.5, 0.5], color_std=[0.5, 0.5, 0.5]):
        self.color_mean = np.array(color_mean)
        self.color_std = np.array(color_std)

    def __call__(self, data):
        if data['feat'][:, :3].max() > 1:
            data['feat'][:, :3] /= 255.
        data['feat'][:, :3] = (
            data['feat'][:, :3] - self.color_mean) / self.color_std
        return data
