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

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from pcseg.cvlibs import manager
from pcseg.models import layers


@manager.MODELS.add_component
class PointNetV2(nn.Layer):
    def __init__(self,
                 in_channels=3,
                 num_classes=13,
                 use_msg=True,
                 n_samples=32,
                 pretrained=None):
        super().__init__()
        if use_msg:
            self.sa1 = PointNetSetAbstractionMSG(1024, [0.05, 0.1], [16, 32],
                                                 in_channels + 3,
                                                 [[16, 16, 32], [32, 32, 64]])
            self.sa2 = PointNetSetAbstractionMSG(256, [0.1, 0.2], [16, 32],
                                                 32 + 64 + 3,
                                                 [[64, 64, 128], [64, 96, 128]])
            self.sa3 = PointNetSetAbstractionMSG(
                64, [0.2, 0.4], [16, 32], 128 + 128 + 3,
                [[128, 196, 256], [128, 196, 256]])
            self.sa4 = PointNetSetAbstractionMSG(
                16, [0.4, 0.8], [16, 32], 256 + 256 + 3,
                [[256, 256, 512], [256, 384, 512]])
            self.fp4 = PointNetFeaturePropagation(512 + 512 + 256 + 256,
                                                  [256, 256])
            self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
            self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
            self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
            self.conv1 = nn.Conv1D(128, 128, 1)
            self.bn1 = nn.BatchNorm1D(128)
            self.drop1 = nn.Dropout(0.5)
            self.conv2 = nn.Conv1D(128, num_classes, 1)
        else:
            self.sa1 = PointNetSetAbstraction(
                1024, 0.1, n_samples, in_channels + 3, [32, 32, 64], False)
            self.sa2 = PointNetSetAbstraction(256, 0.2, n_samples, 64 + 3,
                                              [64, 64, 128], False)
            self.sa3 = PointNetSetAbstraction(64, 0.4, n_samples, 128 + 3,
                                              [128, 128, 256], False)
            self.sa4 = PointNetSetAbstraction(16, 0.8, n_samples, 256 + 3,
                                              [256, 256, 512], False)
            self.fp4 = PointNetFeaturePropagation(768, [256, 256])
            self.fp3 = PointNetFeaturePropagation(384, [256, 256])
            self.fp2 = PointNetFeaturePropagation(320, [256, 128])
            self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
            self.conv1 = nn.Conv1D(128, 128, 1)
            self.bn1 = nn.BatchNorm1D(128)
            self.drop1 = nn.Dropout(0.5)
            self.conv2 = nn.Conv1D(128, num_classes, 1)

    def forward(self, x):
        points = x
        xyz = x[:, :3, :]

        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        return [x]


class PointNetSetAbstraction(nn.Layer):
    def __init__(self,
                 n_points,
                 radius,
                 n_sample,
                 in_channels,
                 out_channel_list,
                 group_all=False):
        super().__init__()
        self.n_points = n_points
        self.radius = radius
        self.n_sample = n_sample
        self.in_channels = in_channels
        self.group_all = group_all

        self.mlp_convs = nn.LayerList()
        self.mlp_bns = nn.LayerList()
        last_channel = in_channels
        for out_channels in out_channel_list:
            self.mlp_convs.append(
                nn.Conv2D(
                    last_channel, out_channels, kernel_size=1))
            self.mlp_bns.append(nn.BatchNorm2D(out_channels))
            last_channel = out_channels

    def forward(self, xyz, points):
        xyz = xyz.transpose([0, 2, 1])
        if points is not None:
            points = points.transpose([0, 2, 1])
        if self.group_all:
            new_xyz, new_points = layers.sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = layers.sample_and_group(
                self.n_points, self.radius, self.n_sample, xyz, points)
        new_points = new_points.transpose([0, 3, 2, 1])
        for i, conv in enumerate(self.mlp_convs):
            new_points = F.relu(self.mlp_bns[i](self.mlp_convs[i](new_points)))
        new_points = paddle.max(new_points, axis=2)
        new_xyz = new_xyz.transpose([0, 2, 1])
        return new_xyz, new_points


class PointNetSetAbstractionMSG(nn.Layer):
    def __init__(self, n_points, radius_list, n_sample_list, in_channel,
                 mlp_dim_list):
        super().__init__()
        self.n_points = n_points
        self.radius_list = radius_list
        self.n_sample_list = n_sample_list
        self.conv_blocks = nn.LayerList()
        self.bn_blocks = nn.LayerList()
        for i in range(len(mlp_dim_list)):
            convs = nn.LayerList()
            bns = nn.LayerList()
            last_channel = in_channel
            for out_channel in mlp_dim_list[i]:
                convs.append(nn.Conv2D(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2D(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        xyz = xyz.transpose([0, 2, 1])
        if points is not None:
            points = points.transpose([0, 2, 1])

        B, N, C = xyz.shape
        S = self.n_points
        new_xyz = layers.index_points(xyz, layers.farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.n_sample_list[i]
            group_idx = layers.query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = layers.index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.reshape([B, S, 1, C])
            if points is not None:
                grouped_points = layers.index_points(points, group_idx)
                grouped_points = paddle.concat(
                    [grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.transpose(
                [0, 3, 2, 1])  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = paddle.max(grouped_points, 2)  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.transpose([0, 2, 1])
        new_points_concat = paddle.concat(new_points_list, axis=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Layer):
    def __init__(self, in_channels, out_channel_list):
        super().__init__()
        self.mlp_convs = nn.LayerList()
        self.mlp_bns = nn.LayerList()
        last_channel = in_channels
        for out_channel in out_channel_list:
            self.mlp_convs.append(nn.Conv1D(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1D(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        inputs:
            xyz1: [B, C, N]
            xyz2: [B, C, S]
            points1: [B, D, N]
            points2: [B, D, S]
        """
        xyz1 = xyz1.transpose([0, 2, 1])
        xyz2 = xyz2.transpose([0, 2, 1])
        points2 = points2.transpose([0, 2, 1])

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = paddle.tile(points2, [1, N, 1])
        else:
            dists = layers.square_distance(xyz1, xyz2)
            idx = dists.argsort(axis=-1)
            dists = dists.sort(axis=-1)

            dists, idx = dists[:, :, :3], idx[:, :, :3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = paddle.sum(dist_recip, axis=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = paddle.sum(layers.index_points(points2, idx) *
                                             weight.reshape([B, N, 3, 1]),
                                             axis=2)

        if points1 is not None:
            points1 = points1.transpose([0, 2, 1])
            new_points = paddle.concat([points1, interpolated_points], axis=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.transpose([0, 2, 1])
        for i, conv in enumerate(self.mlp_convs):
            new_points = F.relu(self.mlp_bns[i](conv(new_points)))
        return new_points
