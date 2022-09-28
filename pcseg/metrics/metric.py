import paddle
import numpy as np


class Metric:
    def __init__(self, num_classes, ignore_index=None):
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.intersection_area = 0
        self.pred_area = 0
        self.label_area = 0

    def update(self, pred, label):
        assert isinstance(
            pred, paddle.
            Tensor), "The logit for Metric must be Tensor, but got {}.".format(
                type(pred))
        assert isinstance(
            label, paddle.
            Tensor), "The label for Metric must be Tensor, but got {}.".format(
                type(pred))
        intersect_area, pred_area, label_area = self.calculate(pred, label)
        self.intersection_area += intersect_area
        self.pred_area += pred_area
        self.label_area += label_area

    def calculate(self, pred, label):
        pred_area = []
        label_area = []
        intersect_area = []
        mask = label != self.ignore_index
        for i in range(self.num_classes):
            pred_i = paddle.logical_and(pred == i, mask)
            label_i = label == i
            intersect_i = paddle.logical_and(pred_i, label_i)
            pred_area.append(paddle.sum(pred_i.astype('int')))
            label_area.append(paddle.sum(label_i.astype('int')))
            intersect_area.append(paddle.sum(intersect_i.astype('int')))
        pred_area = paddle.concat(pred_area)
        label_area = paddle.concat(label_area)
        intersect_area = paddle.concat(intersect_area)
        return intersect_area, pred_area, label_area

    def reset(self):
        self.intersection_area = 0
        self.pred_area = 0
        self.label_area = 0

    def get_metrics(self):
        intersection_area = self.intersection_area.numpy()
        pred_area = self.pred_area.numpy()
        label_area = self.pred_area.numpy()
        union = pred_area + label_area - intersection_area
        class_iou = []
        class_acc = []
        for i in range(len(intersection_area)):
            if i == self.ignore_index:
                continue
            if union[i] == 0:
                iou = 0
            else:
                iou = intersection_area[i] / union[i]
            class_iou.append(iou)
            if pred_area[i] == 0:
                acc = 0
            else:
                acc = intersection_area[i] / pred_area[i]
            class_acc.append(acc)
        macc = np.sum(intersection_area) / np.sum(pred_area)
        miou = np.mean(class_iou)
        return miou, np.array(class_iou), macc, np.array(class_acc)


#
# class Metric:
#     def __init__(self, num_classes, ignore_index=None):
#         self.last_scan_size = None
#         self.ones = None
#         self.conf_matrix = None
#         self.num_classes = num_classes
#         self.ignore_index = paddle.tensor(ignore_index).long()
#         self.include = paddle.tensor(
#             [n for n in range(self.num_classes) if n != self.ignore_index])
#         self.reset()
#
#     def update(self, pred, label):
#         if isinstance(pred, np.ndarray):
#             pred = paddle.to_tensor(np.array(pred))
#         if isinstance(pred, np.ndarray):
#             label = paddle.to_tensor(np.array(label))
#
#         pred_row = pred.reshape(-1)
#         label_row = label.reshape(-1)
#         idxs = paddle.stack([pred_row, label_row], axis=0)
#
#         if self.ones is None or self.last_scan_size != idxs.shape[-1]:
#             self.ones = paddle.ones((idxs.shape[-1]))
#             self.last_scan_size = idxs.shape[-1]
#
#         self.conf_matrix = self.conf_matrix.index_put_(
#             tuple(idxs), self.ones, accumulate=True)
#
#     def reset(self):
#         self.conf_matrix = paddle.zeros(
#             (self.num_classes, self.num_classes))
#         self.ones = None
#         self.last_scan_size = None
#
#     def getStats(self):
#         conf = self.conf_matrix.clone().double()
#         if self.ignore_index is not None:
#             conf[self.ignore_index] = 0
#             conf[:, self.ignore_index] = 0
#
#         tp = conf.diag()
#         fp = conf.sum(axis=1) - tp
#         fn = conf.sum(axis=0) - tp
#         return tp, fp, fn
#
#     def getIoU(self):
#         tp, fp, fn = self.getStats()
#         intersection = tp
#         union = tp + fp + fn + 1e-15
#         iou = intersection / union
#         iou_mean = (intersection[self.include] / union[self.include]).mean()
#         return iou_mean.numpy(), iou.numpy()
#
#     def getacc(self):
#         tp, fp, fn = self.getStats()
#         total_tp = tp.sum()
#         total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
#         acc_mean = total_tp / total
#         return acc_mean.cpu().detach().numpy()
