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
import pickle
import numpy as np
from tqdm import tqdm

from pcseg.datasets import voxelize, crop_pc
from pcseg.cvlibs import manager
from pcseg.transforms import Compose
from pcseg.utils import logger

from paddle.io import Dataset


@manager.DATASETS.add_component
class S3DIS(Dataset):
    num_classes = 13
    """
    S3DIS dataset.
    
    Args:
        dataset_root (str, optional): The path to dataset dir. Default: 'data/s3disfull'.
        val_area (int, optional): The number of area for validation. Default: 5.
        voxel_size (float, optional): The voxel size for donwampling. Default: 0.04.
        voxel_max (None, optional): Subsample the max number of point per point cloud. Set None to use all points. Default: None.
        mode (str, optional): The mode of dataset, support ['train', 'val', 'test']. Default: 'train'.
        transforms (None, optional): The transforms used to augment data. Default: None.
        presample (bool, optional): Whether downsample each point cloud before training. Set to False to downsample on the fly. Default: False.
        use_raw_data (bool, optional): Whether use raw point cloud data. Default: False.
    """

    def __init__(
            self,
            dataset_root='data/s3disfull',
            val_area=5,
            voxel_size=0.04,
            voxel_max=None,
            mode='train',
            transforms=None,
            presample=False,
            use_raw_data=False, ):
        super().__init__()
        self.mode, self.voxel_size, self.voxel_max, self.presample, self.use_raw_data = mode, voxel_size, voxel_max, presample, use_raw_data
        self.transforms = Compose(transforms)
        raw_root = os.path.join(dataset_root, 'raw')
        assert os.path.exists(
            raw_root), "{} not exists, please check your data path.".format(
                raw_root)
        self.raw_root = raw_root

        data_list = sorted(os.listdir(raw_root))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]
        if mode == 'train':
            self.data_list = [
                item for item in data_list
                if not 'Area_{}'.format(val_area) in item
            ]
        else:
            self.data_list = [
                item for item in data_list if 'Area_{}'.format(val_area) in item
            ]

        processed_root = os.path.join(dataset_root, 'processed')
        filename = os.path.join(
            processed_root, f's3dis_{mode}_area{val_area}_{voxel_size:.3f}.pkl')
        print(filename, mode)
        if presample and not os.path.exists(filename):
            np.random.seed(10001)
            self.data = []
            for item in tqdm(
                    self.data_list,
                    desc='Loading S3DISFull {} split on val Area {}'.format(
                        mode, val_area)):
                data_path = os.path.join(raw_root, item + '.npy')
                cdata = np.load(data_path).astype(np.float32)
                cdata[:, :3] -= np.min(cdata[:, :3], 0)
                if voxel_size:
                    coord, feat, label = cdata[:, 0:3], cdata[:, 3:
                                                              6], cdata[:, 6:7]
                    uniq_idx = voxelize(coord, voxel_size)
                    coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[
                        uniq_idx]
                    cdata = np.hstack((coord, feat, label))
                self.data.append(cdata)
            npoints = np.array([len(data) for data in self.data])
            logger.info(
                'mode: {}, median npoints {}, avg num points {}, std {}.'.
                format(self.mode,
                       np.median(npoints), np.average(npoints), np.std(
                           npoints)))
            os.makedirs(processed_root, exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.data, f)
                logger.info("{} saved successfully.".format(filename))
        elif presample:
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
                logger.info("{} load successfully.".format(filename))
        logger.info("Totally {} samples in {} set.".format(
            len(self.data_list), self.mode))

    def __getitem__(self, idx):
        if self.presample:
            coord, feat, label = np.split(self.data[idx], [3, 6], axis=1)
        else:
            data_path = os.path.join(self.raw_root,
                                     self.data_list[idx] + '.npy')
            cdata = np.load(data_path).astype(np.float32)
            cdata[:, :3] -= np.min(cdata[:, :3], 0)
            coord, feat, label = cdata[:, :3], cdata[:, 3:6], cdata[:, 6:7]
            coord, feat, label = crop_pc(
                coord,
                feat,
                label,
                self.mode,
                self.voxel_size,
                self.voxel_max,
                downsample=not self.presample,
                use_raw_data=self.use_raw_data)

        label = label.squeeze(-1).astype(np.compat.long)
        data = {'pos': coord, 'feat': feat, 'label': label}

        if self.transforms is not None:
            data = self.transforms(data)
        # data['x'] = torch.cat((data['x'], torch.from_numpy(
        #     coord[:, 3-self.n_shifted:3].astype(np.float32))), dim=-1)
        return data

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    train_dataset = S3DIS(
        dataset_root="/home/ld/Desktop/pointcloud/PCSeg/data/s3disfull",
        presample=False,
        use_raw_data=False)
    logger.info(len(train_dataset))
    d = train_dataset[0]
    # print(type(d), d.keys())

    val_dataset = S3DIS(
        dataset_root="/home/ld/Desktop/pointcloud/PCSeg/data/s3disfull",
        mode='val',
        presample=True,
        use_raw_data=False)
    logger.info(len(val_dataset))
    d = val_dataset[0]
    # print(type(d), d.keys())
