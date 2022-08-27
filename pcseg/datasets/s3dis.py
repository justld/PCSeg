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

import os
import numpy as np
from tqdm import tqdm

from pcseg.cvlibs import manager
from pcseg.transforms import Compose

from paddle.io import Dataset


@manager.DATASETS.add_component
class S3DIS(Dataset):
    num_classes = 13
    CLASSES = ('ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
               'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter')
    PALETTE = [[0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 0],
               [255, 0, 255], [100, 100, 255], [200, 200, 100],
               [170, 120, 200], [255, 0, 0], [200, 100, 100], [10, 200, 100],
               [200, 200, 200], [50, 50, 50]]
    """
    S3DIS dataset.
    
    Args:
        dataset_root (str, optional): The path to dataset dir. Default: 'data/s3disfull'.
        val_area (int, optional): The number of area for validation. Default: 5.
        num_points (int, optional): The number of sampled points. Default: 4096.
        block_size (float, optional): The block size for sampling. Default: 1.0.
        mode (str, optional): The mode of dataset, support ['train', 'val', 'test']. Default: 'train'.
        transforms (None, optional): The transforms used to augment data. Default: None.
    """

    def __init__(self,
                 dataset_root='data/s3dis/stanford_indoor3d',
                 val_area=5,
                 num_points=4096,
                 block_size=1.0,
                 mode='train',
                 transforms=None):
        super().__init__()
        self.num_point = num_points
        self.block_size = block_size
        self.transforms = Compose(transforms)
        rooms = sorted(os.listdir(dataset_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if mode.lower() == 'train':
            rooms_split = [
                room for room in rooms if not 'Area_{}'.format(val_area) in room
            ]
        else:
            rooms_split = [
                room for room in rooms if 'Area_{}'.format(val_area) in room
            ]

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(S3DIS.num_classes)

        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            room_path = os.path.join(dataset_root, room_name)
            room_data = np.load(room_path)
            points, labels = room_data[:, 0:6], room_data[:, 6]
            tmp, _ = np.histogram(labels, range(14))
            labelweights += tmp
            coord_min, coord_max = np.amin(
                points, axis=0)[:3], np.amax(
                    points, axis=0)[:3]
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(
                coord_max)
            num_point_all.append(labels.size)
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(
            np.amax(labelweights) / labelweights, 1 / 3.0)
        print("Class label weight in S3DIS: ", self.labelweights)
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) / num_points)
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] *
                             int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), mode))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]
        labels = self.room_labels[room_idx]
        N_points = points.shape[0]

        while True:
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [
                self.block_size / 2.0, self.block_size / 2.0, 0
            ]
            block_max = center + [
                self.block_size / 2.0, self.block_size / 2.0, 0
            ]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (
                points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) &
                                  (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(
                point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(
                point_idxs, self.num_point, replace=True)

        selected_points = points[selected_point_idxs, :]
        current_points = np.zeros((self.num_point, 9))
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[
            room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[
            room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[
            room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        # selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        data = {}
        data['pos'] = current_points[:, 0:3]
        data['feat'] = current_points[:, 3:6]
        data['norm_location'] = current_points[:, 6:]
        data['label'] = current_labels
        if self.transforms is not None:
            data = self.transforms(data)
            # current_points, current_labels = self.transform(current_points,
            #                                                 current_labels)
        current_points = np.concatenate(
            [data['pos'], data['feat'], data['norm_location']],
            axis=1).transpose([1, 0])
        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)


@manager.DATASETS.add_component
class ScannetDatasetWholeScene(S3DIS):
    def __init__(self,
                 dataset_root,
                 block_points=4096,
                 mode='val',
                 val_area=5,
                 stride=0.5,
                 block_size=1.0,
                 padding=0.001,
                 transforms=None):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.dataset_root = dataset_root
        self.mode = mode
        self.stride = stride
        self.transforms = Compose(transforms)
        self.scene_points_num = []
        if self.mode == 'train':
            self.file_list = [
                d for d in os.listdir(dataset_root)
                if d.find('Area_%d' % val_area) is -1
            ]
        else:
            self.file_list = [
                d for d in os.listdir(dataset_root)
                if d.find('Area_%d' % val_area) is not -1
            ]
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for file in self.file_list:
            data = np.load(os.path.join(dataset_root, file))
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])
            self.semantic_labels_list.append(data[:, 6])
            coord_min, coord_max = np.amin(
                points, axis=0)[:3], np.amax(
                    points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(
                coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(ScannetDatasetWholeScene.num_classes)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(
                seg, range(ScannetDatasetWholeScene.num_classes + 1))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(
            np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:, :6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(
            points, axis=0)[:3], np.amax(
                points, axis=0)[:3]
        grid_x = int(
            np.ceil(
                float(coord_max[0] - coord_min[0] - self.block_size) /
                self.stride) + 1)
        grid_y = int(
            np.ceil(
                float(coord_max[1] - coord_min[1] - self.block_size) /
                self.stride) + 1)
        data_room, label_room, sample_weight, index_room = np.array(
            []), np.array([]), np.array([]), np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where((points[:, 0] >= s_x - self.padding) & (
                    points[:, 0] <= e_x + self.padding) & (
                        points[:, 1] >= s_y - self.padding) & (
                            points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (
                    point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(
                    point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size /
                                                       2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size /
                                                       2.0)
                # data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                data_room = np.vstack(
                    [data_room, data_batch]) if data_room.size else data_batch
                label_room = np.hstack(
                    [label_room,
                     label_batch]) if label_room.size else label_batch
                sample_weight = np.hstack(
                    [sample_weight,
                     batch_weight]) if label_room.size else batch_weight
                index_room = np.hstack(
                    [index_room, point_idxs]) if index_room.size else point_idxs
        data = {}
        data['pos'] = data_room[:, 0:3]
        data['feat'] = data_room[:, 3:6]
        data['norm_location'] = data_room[:, 6:]
        if self.transforms is not None:
            data = self.transforms(data)
        data_room = np.concatenate(
            [data['pos'], data['feat'], data['norm_location']], axis=1)

        data_room = data_room.reshape(
            (-1, self.block_points, data_room.shape[1])).transpose([0, 2, 1])
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))

        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)
