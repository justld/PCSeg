import argparse
import time
import os

import paddle
from paddle.amp import GradScaler, auto_cast

from pcseg.transforms import Compose, NormalizeRangeImage, LoadSemanticKITTI
from pcseg.dataset import SemanticKITTI
from pcseg.losses import CrossEntropyLoss
from pcseg.models import SqueezeSegV3, DDRNet23_slim
from pcseg.utils import TimeAverager, calculate_eta, resume
from pcseg.core import evaluate

# from pcseg.lr import WarmUpLR


def arg_parse():
    parser = argparse.ArgumentParser('train')
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--epochs', type=int, required=False, default=100)
    parser.add_argument('--num_classes', type=int, required=False, default=20)
    parser.add_argument(
        '--learning_rate', type=float, required=False, default=0.001)
    parser.add_argument(
        '--weight_decay', type=float, required=False, default=0.0001)
    parser.add_argument('--log_iters', type=int, required=False, default=20)
    parser.add_argument('--device', type=str, required=False, default='cuda')
    parser.add_argument(
        '--save_dir', type=str, required=False, default='output')
    parser.add_argument('--save_interval', type=int, required=False, default=1)
    parser.add_argument('--precision', type=str, default='fp16', required=False)
    parser.add_argument('--resume_dir', type=str, default=None, required=False)
    args = parser.parse_args()
    return args


def loss_computation(loss_fn, logits, label):
    loss = 0.0
    for logit in logits:
        if logit.shape[2:] != label.shape[2:]:
            label = paddle.nn.functional.interpolate(
                label, size=logit.shape[2:], mode='nearest')
        loss += loss_fn(logit, label.squeeze(1))
    return loss


def main(args):
    # device = args.device
    trans = Compose([
        LoadSemanticKITTI(
            project_label=True, H=64, W=2048, fov_up=3.0, fov_down=-25.0),
        NormalizeRangeImage(
            mean=[12.12, 10.88, 0.23, -1.04, 0.21],
            std=[12.32, 11.47, 6.91, 0.86, 0.16], )
    ])
    train_dataset = SemanticKITTI(
        root='/home/ld/PycharmProjects/SemanticKitti/SemanticKITTI',
        sequences=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
        mode='train',
        ignore_index=0,
        transforms=trans, )
    train_dataloader = paddle.io.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        drop_last=True, )
    val_dataset = SemanticKITTI(
        root='/home/ld/PycharmProjects/SemanticKitti/SemanticKITTI',
        sequences=[8],
        mode='val',
        ignore_index=0,
        transforms=trans, )
    val_dataloader = paddle.io.DataLoader(
        val_dataset,
        batch_size=1, )

    remap_lut = train_dataset.remap_lut
    content = paddle.zeros((train_dataset.NUM_CLASSES, ))
    for cl, freq in train_dataset.CONTENT.items():
        x_cl = remap_lut[cl]
        content[x_cl] += freq
    weight = 1. / (content + 1e-5)
    if train_dataset.ignore_index in range(train_dataset.NUM_CLASSES):
        weight[train_dataset.ignore_index] = 0.

    loss_fn = CrossEntropyLoss(
        ignore_index=train_dataset.ignore_index, weight=weight)
    # model = SqueezeSegV3(in_channels=5, num_classes=args.num_classes, layers=21)
    model = DDRNet23_slim(in_channels=5, num_classes=args.num_classes)

    lr = paddle.optimizer.lr.PolynomialDecay(
        args.learning_rate,
        decay_steps=len(train_dataloader) * (args.epochs - 1),
        end_lr=0.0001,
        power=1.0,
        cycle=False,
        last_epoch=-1,
        verbose=False)
    lr_scheduler = paddle.optimizer.lr.LinearWarmup(
        lr,
        warmup_steps=len(train_dataloader),
        start_lr=0,
        end_lr=args.learning_rate,
        last_epoch=-1,
        verbose=False)
    optimizer = paddle.optimizer.Momentum(
        parameters=model.parameters(),
        learning_rate=lr_scheduler,
        weight_decay=args.weight_decay)

    if args.resume_dir:
        print('resume model and opt from {}.'.format(args.resume_dir))
        resume(args.resume_dir, model, optimizer)

    best_iou = 0.0
    avg_loss = 0.0
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()

    if args.precision == 'fp16':
        scaler = GradScaler(init_loss_scaling=1024)

    for epoch in range(args.epochs):
        model.train()

        for iter, data in enumerate(train_dataloader):
            reader_cost_averager.record(time.time() - batch_start)
            feat = data['data']
            label = data['label']

            optimizer.clear_grad()

            if args.precision == 'fp16':
                with auto_cast(enable=True):
                    logits = model(feat)
                    loss = loss_computation(loss_fn, logits, label)
                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.minimize(optimizer, scaled)
            else:
                logits = model(feat)
                loss = loss_computation(loss_fn, logits, label)
                loss.backward()
                optimizer.step()

            lr_scheduler.step()

            avg_loss += loss.numpy()[0]
            batch_cost_averager.record(
                time.time() - batch_start, num_samples=args.batch_size)
            if iter % args.log_iters == 0:
                avg_loss /= args.log_iters
                remain_iters = int(args.epochs * len(train_dataloader)) - iter
                avg_train_batch_cost = batch_cost_averager.get_average()
                avg_train_reader_cost = reader_cost_averager.get_average()
                eta = calculate_eta(remain_iters, avg_train_batch_cost)

                print(
                    "[TRAIN] epoch: {}/{}, iter: {}/{}, loss: {:.4f}, lr: {:.6f}, batch_cost: {:.4f}, reader_cost: {:.5f}, ips: {:.4f} samples/sec | ETA {}"
                    .format(epoch + 1, args.epochs, iter,
                            int(args.epochs * len(train_dataloader)), avg_loss,
                            optimizer.get_lr(), avg_train_batch_cost,
                            avg_train_reader_cost,
                            batch_cost_averager.get_ips_average(), eta))
                avg_loss = 0.0
                reader_cost_averager.reset()
                batch_cost_averager.reset()

            batch_start = time.time()

        if epoch % args.save_interval == 0:
            print("evaluate start...")
            miou, iou, macc, acc = evaluate(
                model,
                val_dataloader,
                val_dataset.NUM_CLASSES,
                ignore_index=val_dataset.ignore_index)
            if miou > best_iou:
                best_iou = miou
                save_dir = os.path.join(args.save_dir, str(epoch))
                os.makedirs(save_dir, exist_ok=True)
                paddle.save(model.state_dict(),
                            os.path.join(save_dir, 'model.pth'))
                paddle.save(optimizer.state_dict(),
                            os.path.join(save_dir, 'optimizer.pth'))
            print("EVAL Metric, best_iou: {}, mIOU: {}, macc: {}".format(
                best_iou, miou, macc))
            print("Class Iou: {}".format(iou))
            print("Class Acc: {}".format(acc))


if __name__ == '__main__':
    args = arg_parse()
    main(args)
