import paddle
from tqdm import tqdm

from pcseg.metrics import Metric


def evaluate(model, val_dataloader, num_classes, ignore_index):
    model.eval()
    metric = Metric(num_classes, ignore_index=ignore_index)
    with paddle.no_grad():
        for data in tqdm(val_dataloader):
            feat = data['data']
            label = data['label']
            logit = model(feat)[0]
            pred = logit.argmax(axis=1)

            metric.update(pred, label)
    miou, iou, macc, acc = metric.get_metrics()
    return miou, iou, macc, acc
