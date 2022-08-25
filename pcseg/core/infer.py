# Copyright (c) 2022 jutld Authors. All Rights Reserved.
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

import collections.abc

import numpy as np
import paddle


def inference(model, features):
    """
    Inference for image.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        features (Tensor): the input features.

    Returns:
        Tensor: If ori_shape is not None, a prediction with shape (1, 1, num_points) is returned.
            If ori_shape is None, a logit with shape (1, num_classes, num_points) is returned.
    """
    logits = model(features)
    if not isinstance(logits, collections.abc.Sequence):
        raise TypeError(
            "The type of logits must be one of collections.abc.Sequence, e.g. list, tuple. But received {}"
            .format(type(logits)))
    logit = logits[0]
    pred = paddle.argmax(logit, axis=1, keepdim=True, dtype='int32')

    # label = pred[0].numpy().transpose([1, 0])
    # # print(features.shape, label.shape)
    # tmp = np.concatenate([features[0].transpose([1, 0]).numpy(), label], axis=1)
    # with open('infer.txt', 'w') as f:
    #         for points in tmp:
    #             line = ''
    #             for idx, x in enumerate(points):
    #                 if idx < 3:
    #                     line = line + str(x) + ' '
    #                 elif 3 <= idx < 6:
    #                     line = line + str(int(x)) + ' '
    #                 else:
    #                     line = line + str(int(x)) + '\n'
    #             line += '\n'
    #             f.write(line)

    return pred, logit
