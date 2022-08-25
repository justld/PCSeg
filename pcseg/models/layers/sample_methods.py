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

from abc import ABC, abstractmethod

import paddle
import math
import random


def square_distance(src, dst):
    """
    Input:
        src (Tensor): src points. [B, N, C]
        dst (Tensor): dst points. [B, M, C]
    Return:
        dist: per-point square distance. [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * paddle.matmul(src, dst.transpose([0, 2, 1]))
    dist += paddle.sum(src**2, axis=-1).reshape([B, N, 1])
    dist += paddle.sum(dst**2, axis=-1).reshape([B, 1, M])
    return dist


def index_points(points, idx):
    """
    Input:
        points (Tensor): Input position data. [B, N, C]
        idx (Tensor): sample indexs. [B, S] or [B, S, samples]
    Return:
        new_points: indexed points data. [B, S, C]
    """
    B, N, C = points.shape
    index_shape = idx.shape
    if idx.ndim == 2:
        B, S = idx.shape
    elif idx.ndim == 3:
        idx = idx.reshape([index_shape[0], index_shape[1] * index_shape[2]])
        B, S = idx.shape
    idx_offset = paddle.arange(0, B).reshape((B, 1)) * N
    idx_offset = paddle.tile(idx_offset, [1, S])
    selected_idx = (idx + idx_offset).reshape((B * S, ))
    points = points.reshape([B * N, C])
    new_points = paddle.index_select(points, selected_idx, axis=0)
    new_points = new_points.reshape(tuple(index_shape) + (C, ))
    return new_points


def farthest_point_sample(xyz, n_point):
    """
    Input:
        xyz (Tensor): input points position data. [B, N, 3]
        n_point (int): The number of samples.
    Return:
        centroids: Sampled point cloud index. [B, n_point]
    """
    B, N, C = xyz.shape
    centroids = paddle.zeros([B, n_point], dtype='int64')
    distance = paddle.ones([B, N]) * 1e10
    farthest = paddle.randint(0, N, shape=(B, ))
    idx = paddle.arange(0, B).reshape((B, )) * N
    for i in range(n_point):
        centroids[:, i] = farthest
        xyz = xyz.reshape([B * N, C])
        centroid = paddle.index_select(
            xyz, farthest + idx, axis=0).reshape((B, 1, C))
        xyz = xyz.reshape([B, N, C])
        dist = paddle.sum((xyz - centroid)**2, axis=-1)
        mask = dist < distance
        if i == 0:
            distance = dist
        else:
            distance[mask] = dist[mask]
        farthest = paddle.argmax(distance, axis=-1)
    return centroids


def query_ball_point(radius, n_sample, xyz, new_xyz):
    """
    Input:
        radius (float): region raidus.
        n_sample (int): The number of samples.
        xyz: The position of point cloud data. [B, N, 3]
        new_xyz: query points. [B, S, 3]
    Return:
        group_idx: grouped points index. [B, S, n_sample]
    """
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = paddle.arange(start=0, end=N, dtype='int').reshape([1, 1, N])
    group_idx = paddle.tile(group_idx, [B, S, 1])
    sqr_dist = square_distance(new_xyz, xyz)
    group_idx[sqr_dist > radius**2] = N
    group_idx = group_idx.sort(axis=-1)[:, :, :n_sample]
    group_first = group_idx[:, :, 0].reshape([B, S, 1])
    group_first = paddle.tile(group_first, [1, 1, n_sample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(n_points, radius, n_sample, xyz, points,
                     returen_fps=False):
    """
    Inputs:
        n_points (int): The number of points in data.
        radius (float): The radius of sampled ball.
        n_sample (int): The number of sampled points.
        xyz (Tensor): The points position data. [B, N, 3]
        points (Tensor): The feature of points. [B, N, D]
        returen_fps (bool, optional): Whether return sample idx.
    Return:
        new_xyz (Tensor): sampled points position data. [B, n_points, n_sample, 3]
        new_points (Tensor): sampled points data. [B, n_points, n_sample, 3 + D]
    """
    # print(n_points, radius, n_sample)
    B, N, C = xyz.shape
    S = n_points
    fps_idx = farthest_point_sample(xyz, n_points)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, n_sample, xyz, new_xyz)

    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.reshape([B, S, 1, C])

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = paddle.concat([grouped_xyz_norm, grouped_points], axis=-1)
    else:
        new_points = grouped_xyz_norm
    if returen_fps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz (Tensor): input points position data. [B, 3, N]
        points (Tensor): input points features. [B, N, D]
    Return:
        new_xyz (Tensor): sampled points position data. [B, 1, 3]
        new_points (Tensor): sampled points data. [B, 1, N, 3 + D]
    """
    B, C, N = xyz.shape
    B, D, N = points.shape
    new_xyz = paddle.zeros([B, 1, C], dtype='float32')
    grouped_xyz = xyz.reshape([B, 1, C, N])
    if points is not None:
        new_points = paddle.concat(
            [grouped_xyz, points.reshape([B, 1, D, N])], axis=2)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points
