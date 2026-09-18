import os
import time

import numpy as np
import cv2
import torch

from .nets import S3FDNet
from .box_utils import nms_


# ============================================================
# IMPORTANT
# ============================================================
#
# The original TalkNet repository contains old code that
# automatically executes:
#
#     gdown --id ...
#
# when the S3FD package is imported.
#
# We DO NOT want that.
#
# Our project already has the downloaded:
#
#     weights/sfd_face.pth
#
# Therefore this package must only contain the model class.
# ============================================================


IMG_MEAN = (
    np.array(
        [
            104.0,
            117.0,
            123.0,
        ]
    )
    [:, np.newaxis, np.newaxis]
    .astype(
        "float32"
    )
)


class S3FD:

    def __init__(
        self,
        device="cpu",
        weight_path=None,
    ):

        self.device = device

        if weight_path is None:

            project_root = (
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(
                            __file__
                        ),
                        "..",
                        "..",
                        "..",
                        "..",
                    )
                )
            )

            weight_path = os.path.join(
                project_root,
                "weights",
                "sfd_face.pth",
            )

        if not os.path.isfile(
            weight_path
        ):

            raise FileNotFoundError(
                "S3FD weights not found:\n"
                f"{weight_path}"
            )

        self.net = (
            S3FDNet(
                device=self.device
            )
            .to(self.device)
        )

        state_dict = torch.load(
            weight_path,
            map_location=self.device,
        )

        self.net.load_state_dict(
            state_dict
        )

        self.net.eval()

    def detect_faces(
        self,
        image,
        conf_th=0.8,
        scales=(1,),
    ):

        width = image.shape[1]

        height = image.shape[0]

        bboxes = np.empty(
            shape=(0, 5)
        )

        with torch.no_grad():

            for scale_factor in scales:

                scaled_img = cv2.resize(
                    image,
                    dsize=(0, 0),
                    fx=scale_factor,
                    fy=scale_factor,
                    interpolation=cv2.INTER_LINEAR,
                )

                scaled_img = np.swapaxes(
                    scaled_img,
                    1,
                    2,
                )

                scaled_img = np.swapaxes(
                    scaled_img,
                    1,
                    0,
                )

                scaled_img = scaled_img[
                    [2, 1, 0],
                    :,
                    :,
                ]

                scaled_img = (
                    scaled_img
                    .astype(
                        "float32"
                    )
                )

                scaled_img -= IMG_MEAN

                scaled_img = scaled_img[
                    [2, 1, 0],
                    :,
                    :,
                ]

                x = (
                    torch.from_numpy(
                        scaled_img
                    )
                    .unsqueeze(0)
                    .to(self.device)
                )

                y = self.net(x)

                detections = y.data

                scale = torch.Tensor(
                    [
                        width,
                        height,
                        width,
                        height,
                    ]
                )

                for i in range(
                    detections.size(1)
                ):

                    j = 0

                    while (
                        j
                        < detections.size(2)
                        and
                        detections[
                            0,
                            i,
                            j,
                            0
                        ]
                        > conf_th
                    ):

                        score = (
                            detections[
                                0,
                                i,
                                j,
                                0
                            ]
                        )

                        pt = (
                            detections[
                                0,
                                i,
                                j,
                                1:
                            ]
                            * scale
                        ).cpu().numpy()

                        pt /= scale_factor

                        bbox = (
                            pt[0],
                            pt[1],
                            pt[2],
                            pt[3],
                            score.item(),
                        )

                        bboxes = (
                            np.vstack(
                                [
                                    bboxes,
                                    bbox,
                                ]
                            )
                        )

                        j += 1

        if len(bboxes) > 0:

            keep = nms_(
                bboxes,
                0.1,
            )

            bboxes = bboxes[
                keep
            ]

        return bboxes