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

import numpy as np
import time
import paddle
import paddle.nn.functional as F

from pcseg.utils import metrics, TimeAverager, logger, progbar
from pcseg.core import infer

np.set_printoptions(suppress=True)


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b,
                                                                     n])] += 1
    return vote_label_pool


def visual_points(output_dir, file_name, points, gt, pred, color_palette):
    gt_f = open(os.path.join(output_dir, file_name + '_gt.txt'), 'w')
    pred_f = open(os.path.join(output_dir, file_name + '_pred.txt'), 'w')
    for i in range(points.shape[0]):
        gt_f.write("{} {} {} {} {} {}\n".format(
            points[i, 0],
            points[i, 1],
            points[i, 2],
            color_palette[gt[i]][0],
            color_palette[gt[i]][1],
            color_palette[gt[i]][2], ))
        pred_f.write("{} {} {} {} {} {}\n".format(
            points[i, 0],
            points[i, 1],
            points[i, 2],
            color_palette[pred[i]][0],
            color_palette[pred[i]][1],
            color_palette[pred[i]][2], ))
    gt_f.close()
    pred_f.close()


def evaluate(model,
             eval_dataset,
             precision='fp32',
             amp_level='O1',
             num_workers=0,
             print_detail=True,
             auc_roc=False,
             num_votes=3):
    """
    Launch evalution.

    Args:
        model（nn.Layer): A sementic segmentation model.
        eval_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        precision (str, optional): Use AMP if precision='fp16'. If precision='fp32', the evaluation is normal.
        amp_level (str, optional): Auto mixed precision level. Accepted values are “O1” and “O2”: O1 represent mixed precision, the input data type of each operator will be casted by white_list and black_list; O2 represent Pure fp16, all operators parameters and input data will be casted to fp16, except operators in black_list, don’t support fp16 kernel and batchnorm. Default is O1(amp)
        num_workers (int, optional): Num workers for data loader. Default: 0.
        print_detail (bool, optional): Whether to print detailed information about the evaluation process. Default: True.
        auc_roc(bool, optional): whether add auc_roc metric

    Returns:
        float: The mIoU of validation datasets.
        float: The accuracy of validation datasets.
    """
    model.eval()
    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
    if nranks > 1:
        # Initialize parallel environment if not done.
        if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
        ):
            paddle.distributed.init_parallel_env()
    batch_sampler = paddle.io.DistributedBatchSampler(
        eval_dataset, batch_size=1, shuffle=False, drop_last=False)
    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True, )

    total_iters = len(loader)
    intersect_area_all = paddle.zeros([1], dtype='int64')
    pred_area_all = paddle.zeros([1], dtype='int64')
    label_area_all = paddle.zeros([1], dtype='int64')
    logits_all = None
    label_all = None

    if print_detail:
        logger.info("Start evaluating (total_samples: {}, total_iters: {})...".
                    format(len(eval_dataset), total_iters))

    progbar_val = progbar.Progbar(
        target=total_iters, verbose=1 if nranks < 2 else 2)
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    with paddle.no_grad():
        for iter, data in enumerate(loader):
            reader_cost_averager.record(time.time() - batch_start)
            BATCH_SIZE = 16
            scene_data, scene_label, scene_smpw, scene_point_index = data[
                0], data[1], data[2], data[3]
            scene_data = scene_data[0].astype('float32')
            label = eval_dataset.semantic_labels_list[iter]
            scene_point_index = scene_point_index[0].numpy()
            scene_smpw = scene_smpw[0].numpy()

            vote_label_pool = np.zeros(
                (eval_dataset.semantic_labels_list[iter].shape[0],
                 eval_dataset.num_classes))
            for _ in range(num_votes):
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE

                for s_batch in range(s_batch_num):
                    start_idx = s_batch * BATCH_SIZE
                    end_idx = min((s_batch + 1) * BATCH_SIZE, num_blocks)
                    real_batch_size = end_idx - start_idx

                    batch_data = scene_data[start_idx:end_idx, ...]
                    batch_point_index = scene_point_index[start_idx:end_idx]
                    batch_smpw = scene_smpw[start_idx:end_idx, ...]

                    if precision == 'fp16':
                        with paddle.amp.auto_cast(
                                level=amp_level,
                                enable=True,
                                custom_white_list={
                                    "elementwise_add", "batch_norm",
                                    "sync_batch_norm"
                                },
                                custom_black_list={'bilinear_interp_v2'}):
                            pred, logits = infer.inference(model, batch_data)
                    else:
                        pred, logits = infer.inference(model, batch_data)
                    vote_label_pool = add_vote(
                        vote_label_pool, batch_point_index[0:real_batch_size],
                        pred[0:real_batch_size].squeeze(1).numpy(),
                        batch_smpw[0:real_batch_size])
            pred = np.argmax(vote_label_pool, 1)

            # visual
            output_dir = 'output/visual'
            os.makedirs(output_dir, exist_ok=True)
            visual_points(
                output_dir, eval_dataset.file_list[iter][:-4],
                eval_dataset.scene_points_list[iter],
                eval_dataset.semantic_labels_list[iter].astype('int').tolist(),
                pred.tolist(), eval_dataset.PALETTE)

            intersect_area, pred_area, label_area = metrics.calculate_area(
                paddle.to_tensor(pred),
                paddle.to_tensor(label.reshape((-1, ))),
                eval_dataset.num_classes)

            # Gather from all ranks
            if nranks > 1:
                intersect_area_list = []
                pred_area_list = []
                label_area_list = []
                paddle.distributed.all_gather(intersect_area_list,
                                              intersect_area)
                paddle.distributed.all_gather(pred_area_list, pred_area)
                paddle.distributed.all_gather(label_area_list, label_area)

                if (iter + 1) * nranks > len(eval_dataset):
                    valid = len(eval_dataset) - iter * nranks
                    intersect_area_list = intersect_area_list[:valid]
                    pred_area_list = pred_area_list[:valid]
                    label_area_list = label_area_list[:valid]

                for i in range(len(intersect_area_list)):
                    intersect_area_all = intersect_area_all + intersect_area_list[
                        i]
                    pred_area_all = pred_area_all + pred_area_list[i]
                    label_area_all = label_area_all + label_area_list[i]
            else:
                intersect_area_all = intersect_area_all + intersect_area
                pred_area_all = pred_area_all + pred_area
                label_area_all = label_area_all + label_area

                if auc_roc:
                    logits = F.softmax(logits, axis=1)
                    if logits_all is None:
                        logits_all = logits.numpy()
                        label_all = label.numpy()
                    else:
                        logits_all = np.concatenate(
                            [logits_all, logits.numpy()])  # (KN, C, H, W)
                        label_all = np.concatenate([label_all, label.numpy()])

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(label))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            if local_rank == 0 and print_detail:
                progbar_val.update(iter + 1, [('batch_cost', batch_cost),
                                              ('reader cost', reader_cost)])
            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

    metrics_input = (intersect_area_all, pred_area_all, label_area_all)
    class_iou, miou = metrics.mean_iou(*metrics_input)
    acc, class_precision, class_recall = metrics.class_measurement(
        *metrics_input)
    kappa = metrics.kappa(*metrics_input)
    class_dice, mdice = metrics.dice(*metrics_input)

    if auc_roc:
        auc_roc = metrics.auc_roc(
            logits_all, label_all, num_classes=eval_dataset.num_classes)
        auc_infor = ' Auc_roc: {:.4f}'.format(auc_roc)

    if print_detail:
        infor = "[EVAL] #Images: {} mIoU: {:.4f} Acc: {:.4f} Kappa: {:.4f} Dice: {:.4f}".format(
            len(eval_dataset), miou, acc, kappa, mdice)
        infor = infor + auc_infor if auc_roc else infor
        logger.info(infor)
        logger.info("[EVAL] Class IoU: \n" + str(np.round(class_iou, 4)))
        logger.info("[EVAL] Class Precision: \n" + str(
            np.round(class_precision, 4)))
        logger.info("[EVAL] Class Recall: \n" + str(np.round(class_recall, 4)))
    return miou, acc, class_iou, class_precision, kappa
