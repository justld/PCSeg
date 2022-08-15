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


@manager.LOSSES.add_component
class CrossEntropyLoss(nn.Layer):
    def __init__(self, label_smoothing=None):
        super().__init__()
        self.smoothing = label_smoothing

    def forward(self, logit, label):
        logit = logit.transpose([0, 2, 1])
        if self.smoothing:
            label = F.one_hot(label, logit.shape[1])
            label = F.label_smooth(label, epsilon=self.smoothing)
            x = -F.log_softmax(logit, axis=-1)
            loss = paddle.sum(x * label, axis=-1)
        else:
            loss = F.cross_entropy(logit, label)
        return loss.mean()
