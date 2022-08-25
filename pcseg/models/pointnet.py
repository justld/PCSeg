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


@manager.MODELS.add_component
class PointNet(nn.Layer):
    def __init__(self,
                 in_channels=3,
                 num_classes=13,
                 input_transform=True,
                 feature_transform=True,
                 pretrained=None):
        super().__init__()
        self.feature_transform = feature_transform
        self.stn = STN3d(in_channels) if input_transform else None
        self.conv0_1 = nn.Conv1D(in_channels, 64, 1)
        self.conv0_2 = nn.Conv1D(64, 64, 1)

        self.conv1 = nn.Conv1D(64, 64, 1)
        self.conv2 = nn.Conv1D(64, 128, 1)
        self.conv3 = nn.Conv1D(128, 256, 1)
        self.bn0_1 = nn.BatchNorm1D(64)
        self.bn0_2 = nn.BatchNorm1D(64)
        self.bn1 = nn.BatchNorm1D(64)
        self.bn2 = nn.BatchNorm1D(128)
        self.bn3 = nn.BatchNorm1D(256)
        self.fstn = STNkd(k=64) if feature_transform else None
        self.out_channels = 256 + 64

        self.head = Decoder(self.out_channels, num_classes)

    def forward(self, x):

        B, D, N = x.shape
        if self.stn is not None:
            trans = self.stn(x)
            x = x.transpose([0, 2, 1])
            if D > 3:
                feature = x[:, :, 3:]
                x = x[:, :, :3]
            x = paddle.bmm(x, trans)
            if D > 3:
                x = paddle.concat([x, feature], axis=2)
            x = x.transpose([0, 2, 1])
        x = F.relu(self.bn0_1(self.conv0_1(x)))
        x = F.relu(self.bn0_2(self.conv0_2(x)))

        if self.fstn is not None:
            trans_feat = self.fstn(x)
            x = x.transpose([0, 2, 1])
            x = paddle.bmm(x, trans_feat)
            x = x.transpose([0, 2, 1])
        else:
            trans_feat = None
        point_feat = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = paddle.max(x, 2, keepdim=True)
        # x = x.reshape([B, 1024, 1])
        x = paddle.tile(x, [1, 1, N])
        x = paddle.concat([x, point_feat], axis=1)
        logit = self.head(x)
        if self.training and self.feature_transform:
            return [logit, trans_feat]
        return [logit]


class Decoder(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1D(in_channels, 512, 1)
        self.conv2 = nn.Conv1D(512, 256, 1)
        self.conv3 = nn.Conv1D(256, 128, 1)
        self.conv4 = nn.Conv1D(128, out_channels, 1)

        self.bn1 = nn.BatchNorm1D(512)
        self.bn2 = nn.BatchNorm1D(256)
        self.bn3 = nn.BatchNorm1D(128)

    def forward(self, x):
        b, _, n = x.shape
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x


class STN3d(nn.Layer):
    def __init__(self, channel=3):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1D(channel, 64, 1)
        self.conv2 = nn.Conv1D(64, 128, 1)
        self.conv3 = nn.Conv1D(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1D(64)
        self.bn2 = nn.BatchNorm1D(128)
        self.bn3 = nn.BatchNorm1D(1024)
        self.bn4 = nn.BatchNorm1D(512)
        self.bn5 = nn.BatchNorm1D(256)

        self.iden = paddle.to_tensor(
            np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype('float32')).reshape(
                [1, 9])

    def forward(self, x):
        batchsize = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = paddle.max(x, 2, keepdim=True)
        x = x.reshape([batchsize, 1024])

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = paddle.tile(self.iden, [batchsize, 1])
        x = x + iden
        x = x.reshape([batchsize, 3, 3])
        return x


class STNkd(nn.Layer):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1D(k, 64, 1)
        self.conv2 = nn.Conv1D(64, 128, 1)
        self.conv3 = nn.Conv1D(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1D(64)
        self.bn2 = nn.BatchNorm1D(128)
        self.bn3 = nn.BatchNorm1D(1024)
        self.bn4 = nn.BatchNorm1D(512)
        self.bn5 = nn.BatchNorm1D(256)

        self.k = k
        self.iden = paddle.to_tensor(
            np.eye(self.k).flatten().astype('float32')).reshape(
                [1, self.k * self.k])

    def forward(self, x):
        batchsize = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = paddle.max(x, 2, keepdim=True)
        x = x.reshape([batchsize, 1024])

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = paddle.tile(self.iden, [batchsize, 1])

        x = x + iden
        x = x.reshape([batchsize, self.k, self.k])
        return x
