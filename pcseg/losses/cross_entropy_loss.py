import paddle
import paddle.nn as nn


class CrossEntropyLoss(nn.Layer):
    def __init__(self, weight=None, ignore_index=-255):
        super().__init__()
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, logit, label):
        logit = logit.transpose([0, 2, 3, 1])
        loss = nn.functional.cross_entropy(
            logit, label, weight=self.weight, ignore_index=self.ignore_index)
        return loss
