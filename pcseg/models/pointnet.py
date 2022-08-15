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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from pcseg.cvlibs import manager


@manager.MODELS.add_component
class PointNet(nn.Layer):
    def __init__(self, in_channels=3, num_classes=13, pretrained=None):
        super().__init__()
        self.conv = nn.Conv1D(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return [x]
