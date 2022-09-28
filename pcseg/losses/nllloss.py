# import torch
# import torch.nn as nn
#
#
# class NLLLoss(nn.Module):
#     def __init__(self, weight=None, ignore_index=-255):
#         super().__init__()
#         self.ignore_index = ignore_index
#         self.weight = weight
#
#     def forward(self, logit, label):
#         logit = torch.nn.functional.log_softmax(logit, dim=1)
#         loss = nn.functional.nll_loss(logit, label, weight=self.weight, ignore_index=self.ignore_index)
#         return loss
