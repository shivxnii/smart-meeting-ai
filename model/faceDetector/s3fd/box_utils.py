import numpy as np
from itertools import product

import torch


def nms_(dets, thresh):
    """
    CPU NumPy NMS used by the Smart Meeting S3FD wrapper.

    Returns integer indices of boxes to keep.
    Compatible with NumPy 1.26+.
    """

    if dets is None or len(dets) == 0:
        return np.empty((0,), dtype=np.int64)

    dets = np.asarray(dets, dtype=np.float32)

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = np.maximum(
        0.0,
        x2 - x1,
    ) * np.maximum(
        0.0,
        y2 - y1,
    )

    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:

        i = int(order[0])
        keep.append(i)

        if order.size == 1:
            break

        rest = order[1:]

        xx1 = np.maximum(
            x1[i],
            x1[rest],
        )

        yy1 = np.maximum(
            y1[i],
            y1[rest],
        )

        xx2 = np.minimum(
            x2[i],
            x2[rest],
        )

        yy2 = np.minimum(
            y2[i],
            y2[rest],
        )

        w = np.maximum(
            0.0,
            xx2 - xx1,
        )

        h = np.maximum(
            0.0,
            yy2 - yy1,
        )

        inter = w * h

        union = (
            areas[i]
            + areas[rest]
            - inter
        )

        ovr = np.zeros_like(inter)

        valid = union > 0

        ovr[valid] = (
            inter[valid]
            / union[valid]
        )

        inds = np.where(
            ovr <= thresh
        )[0]

        order = rest[inds]

    # IMPORTANT:
    # np.int was removed from NumPy.
    return np.asarray(
        keep,
        dtype=np.int64,
    )


def decode(
    loc,
    priors,
    variances,
):
    """
    Decode locations from predictions.
    """

    boxes = torch.cat(
        (
            priors[:, :2]
            + loc[:, :2]
            * variances[0]
            * priors[:, 2:],

            priors[:, 2:]
            * torch.exp(
                loc[:, 2:]
                * variances[1]
            ),
        ),
        1,
    )

    boxes[:, :2] -= (
        boxes[:, 2:]
        / 2
    )

    boxes[:, 2:] += (
        boxes[:, :2]
    )

    return boxes


def nms(
    boxes,
    scores,
    overlap=0.5,
    top_k=200,
):
    """
    Torch NMS used by the original S3FD detector.
    """

    keep = scores.new(
        scores.size(0)
    ).zero_().long()

    if boxes.numel() == 0:
        return keep, 0

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = torch.mul(
        x2 - x1,
        y2 - y1,
    )

    _, idx = scores.sort(0)

    idx = idx[
        -top_k:
    ]

    xx1 = boxes.new_empty(
        idx.size(0)
    )

    yy1 = boxes.new_empty(
        idx.size(0)
    )

    xx2 = boxes.new_empty(
        idx.size(0)
    )

    yy2 = boxes.new_empty(
        idx.size(0)
    )

    count = 0

    while idx.numel() > 0:

        i = idx[-1]

        keep[count] = i
        count += 1

        if idx.size(0) == 1:
            break

        idx = idx[:-1]

        xx1 = torch.index_select(
            x1,
            0,
            idx,
        )

        yy1 = torch.index_select(
            y1,
            0,
            idx,
        )

        xx2 = torch.index_select(
            x2,
            0,
            idx,
        )

        yy2 = torch.index_select(
            y2,
            0,
            idx,
        )

        xx1 = torch.clamp(
            xx1,
            min=x1[i],
        )

        yy1 = torch.clamp(
            yy1,
            min=y1[i],
        )

        xx2 = torch.clamp(
            xx2,
            max=x2[i],
        )

        yy2 = torch.clamp(
            yy2,
            max=y2[i],
        )

        w = torch.clamp(
            xx2 - xx1,
            min=0.0,
        )

        h = torch.clamp(
            yy2 - yy1,
            min=0.0,
        )

        inter = w * h

        rem_areas = torch.index_select(
            area,
            0,
            idx,
        )

        union = (
            rem_areas
            - inter
            + area[i]
        )

        iou = torch.zeros_like(
            inter
        )

        valid = union > 0

        iou[valid] = (
            inter[valid]
            / union[valid]
        )

        idx = idx[
            iou.le(overlap)
        ]

    return keep, count


class Detect:

    def __init__(
        self,
        num_classes=2,
        top_k=750,
        nms_thresh=0.3,
        conf_thresh=0.05,
        variance=[0.1, 0.2],
        nms_top_k=5000,
    ):

        self.num_classes = num_classes
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.variance = variance
        self.nms_top_k = nms_top_k

    def forward(
        self,
        loc_data,
        conf_data,
        prior_data,
    ):

        num = loc_data.size(0)

        num_priors = (
            prior_data.size(0)
        )

        conf_preds = (
            conf_data
            .view(
                num,
                num_priors,
                self.num_classes,
            )
            .transpose(2, 1)
        )

        batch_priors = (
            prior_data
            .view(
                -1,
                num_priors,
                4,
            )
            .expand(
                num,
                num_priors,
                4,
            )
        )

        batch_priors = (
            batch_priors
            .contiguous()
            .view(-1, 4)
        )

        decoded_boxes = decode(
            loc_data
            .view(-1, 4),
            batch_priors,
            self.variance,
        )

        decoded_boxes = (
            decoded_boxes
            .view(
                num,
                num_priors,
                4,
            )
        )

        output = torch.zeros(
            num,
            self.num_classes,
            self.top_k,
            5,
            device=loc_data.device,
        )

        for i in range(num):

            boxes = (
                decoded_boxes[i]
                .clone()
            )

            conf_scores = (
                conf_preds[i]
                .clone()
            )

            for cl in range(
                1,
                self.num_classes,
            ):

                c_mask = (
                    conf_scores[cl]
                    .gt(
                        self.conf_thresh
                    )
                )

                scores = (
                    conf_scores[cl][c_mask]
                )

                if scores.numel() == 0:
                    continue

                l_mask = (
                    c_mask
                    .unsqueeze(1)
                    .expand_as(boxes)
                )

                boxes_ = (
                    boxes[l_mask]
                    .view(-1, 4)
                )

                ids, count = nms(
                    boxes_,
                    scores,
                    self.nms_thresh,
                    self.nms_top_k,
                )

                count = min(
                    count,
                    self.top_k,
                )

                if count > 0:

                    output[
                        i,
                        cl,
                        :count,
                    ] = torch.cat(
                        (
                            scores[
                                ids[:count]
                            ].unsqueeze(1),

                            boxes_[
                                ids[:count]
                            ],
                        ),
                        1,
                    )

        return output


class PriorBox:

    def __init__(
        self,
        input_size,
        feature_maps,
        variance=[0.1, 0.2],
        min_sizes=[
            16,
            32,
            64,
            128,
            256,
            512,
        ],
        steps=[
            4,
            8,
            16,
            32,
            64,
            128,
        ],
        clip=False,
    ):

        self.imh = input_size[0]
        self.imw = input_size[1]

        self.feature_maps = feature_maps

        self.variance = variance
        self.min_sizes = min_sizes
        self.steps = steps
        self.clip = clip

    def forward(self):

        mean = []

        for k, fmap in enumerate(
            self.feature_maps
        ):

            feath = fmap[0]
            featw = fmap[1]

            for i, j in product(
                range(feath),
                range(featw),
            ):

                f_kw = (
                    self.imw
                    / self.steps[k]
                )

                f_kh = (
                    self.imh
                    / self.steps[k]
                )

                cx = (
                    j + 0.5
                ) / f_kw

                cy = (
                    i + 0.5
                ) / f_kh

                s_kw = (
                    self.min_sizes[k]
                    / self.imw
                )

                s_kh = (
                    self.min_sizes[k]
                    / self.imh
                )

                mean += [
                    cx,
                    cy,
                    s_kw,
                    s_kh,
                ]

        output = torch.FloatTensor(
            mean
        ).view(-1, 4)

        if self.clip:
            output.clamp_(
                max=1,
                min=0,
            )

        return output