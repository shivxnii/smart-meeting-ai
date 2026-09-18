AUDIO_SAMPLE_RATE = 16000
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import python_speech_features


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

TALKNET_DIR = (
    BASE_DIR / "TalkNet-ASD-main"
)

if str(TALKNET_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(TALKNET_DIR),
    )


# ============================================================
# ORIGINAL TALKNET MODEL
# ============================================================

from model.talkNetModel import (
    talkNetModel,
)

from loss import lossAV

from model.faceDetector.s3fd.nets import (
    S3FDNet,
)

from model.faceDetector.s3fd.box_utils import (
    nms_,
)


# ============================================================
# CONFIG
# ============================================================

from smart_meeting.config import (
    DEVICE,
    TALKNET_MODEL_PATH,
    S3FD_MODEL_PATH,
    TALKNET_FACE_SIZE,
    MIN_FACE_SIZE,
    DEBUG,
)


# ============================================================
# DEVICE
# ============================================================

TORCH_DEVICE = torch.device(
    DEVICE
)

print(
    f"[TalkNet] Device: {TORCH_DEVICE}"
)


# ============================================================
# S3FD FACE DETECTOR
# ============================================================

IMG_MEAN = np.array(
    [104.0, 117.0, 123.0],
    dtype=np.float32,
).reshape(
    3,
    1,
    1,
)


class S3FDDetector:

    def __init__(
        self,
        weights_path=S3FD_MODEL_PATH,
        device=TORCH_DEVICE,
    ):

        self.device = torch.device(
            device
        )

        weights_path = str(
            weights_path
        )

        print(
            "[S3FD] Initializing "
            "face detector..."
        )

        if not os.path.isfile(
            weights_path
        ):

            raise FileNotFoundError(
                "\n"
                "[S3FD] Missing weights:\n"
                f"{weights_path}\n\n"
                "Expected file:\n"
                "weights/sfd_face.pth"
            )

        self.net = S3FDNet(
            device=str(
                self.device
            )
        ).to(
            self.device
        )

        checkpoint = torch.load(
            weights_path,
            map_location=self.device,
        )

        if (
            isinstance(
                checkpoint,
                dict,
            )
            and
            "state_dict" in checkpoint
        ):

            checkpoint = (
                checkpoint[
                    "state_dict"
                ]
            )

        cleaned = {}

        for key, value in (
            checkpoint.items()
        ):

            cleaned[
                key.replace(
                    "module.",
                    "",
                )
            ] = value

        missing, unexpected = (
            self.net.load_state_dict(
                cleaned,
                strict=False,
            )
        )

        if missing:

            raise RuntimeError(
                "[S3FD] Missing model "
                "parameters:\n"
                + "\n".join(
                    missing[:20]
                )
            )

        if unexpected:

            print(
                "[S3FD] Warning: "
                f"{len(unexpected)} "
                "unexpected parameters."
            )

        self.net.eval()

        print(
            "[S3FD] Loaded successfully "
            f"on {self.device}"
        )

        # ----------------------------------------------------
        # NEW: Haar cascade cross-check.
        #
        # S3FD alone can be fooled by textured surfaces
        # (pillow prints, blankets, paper edges) into
        # reporting them as high-confidence faces — raising
        # FACE_DET_CONF / MIN_FACE_SIZE doesn't fix that,
        # because the confidence itself is genuinely high.
        #
        # Haar cascade is a completely different, classical
        # algorithm that looks for actual eye/nose/mouth
        # structure. A box only survives if BOTH detectors
        # agree it's a face. A pillow pattern can fool S3FD,
        # but it has no eye/nose structure, so Haar rejects
        # it — this is the "double-check" mentioned earlier
        # that hadn't actually been wired in yet.
        # ----------------------------------------------------

        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.haar_cascade = (
            cv2.CascadeClassifier(
                cascade_path
            )
        )

        if self.haar_cascade.empty():

            print(
                "[S3FD] WARNING: Haar cascade "
                "failed to load from "
                f"{cascade_path} — false-positive "
                "cross-check is DISABLED. False "
                "detections on textured surfaces "
                "may reappear."
            )

            self.haar_cascade = None

        else:

            print(
                "[S3FD] Haar cascade cross-check "
                "loaded — non-face boxes (pillow/"
                "blanket/paper patterns) will now "
                "be rejected even if S3FD scores "
                "them as high confidence."
            )

        # Throttled debug counter — prints a limited number of
        # times so we can see WHY boxes are being rejected
        # (aspect ratio / size / Haar) without flooding the
        # terminal on every frame.
        self._debug_print_count = 0
        self._debug_print_limit = 30

    @torch.inference_mode()
    def detect_faces(
        self,
        image_rgb,
        conf_th=0.70,
        scales=(0.50,),
    ):

        if (
            image_rgb is None
            or
            image_rgb.size == 0
        ):

            return np.empty(
                (0, 5),
                dtype=np.float32,
            )

        original_h, original_w = (
            image_rgb.shape[:2]
        )

        # Full-resolution grayscale copy, used later for the
        # Haar cascade cross-check. Box coordinates below are
        # already converted back into this original-frame
        # pixel space, so we crop directly from here (no
        # re-scaling needed).
        gray_full = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2GRAY,
        )

        all_boxes = []

        for scale_factor in scales:

            scaled_img = cv2.resize(
                image_rgb,
                dsize=(0, 0),
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_LINEAR,
            )

            # HWC -> CHW
            img = (
                scaled_img
                .transpose(
                    2,
                    0,
                    1,
                )
            )

            img = img[
                [2, 1, 0],
                :,
                :,
            ]

            img = img.astype(
                np.float32
            )

            img -= IMG_MEAN

            img = img[
                [2, 1, 0],
                :,
                :,
            ]

            tensor = torch.from_numpy(
                img
            ).unsqueeze(
                0
            ).to(
                self.device
            )

            output = self.net(
                tensor
            )

            detections = output.data

            # --------------------------------------------------
            # FIX (this was the bounding-box offset bug):
            #
            # detections[...,1:5] are normalized (0-1) coords
            # relative to the network's INPUT, which is
            # scaled_img (original * scale_factor).
            #
            # Multiplying by (original_w, original_h) already
            # converts them directly into ORIGINAL-frame pixel
            # coordinates. No further division by scale_factor
            # is needed or correct.
            #
            # The old code did BOTH: multiplied by original
            # dims AND divided by scale_factor afterwards,
            # which silently inflated every coordinate (with
            # scale_factor=0.5, every coord got doubled),
            # pushing boxes down/right of the real face and
            # onto walls, furniture, bodies.
            # --------------------------------------------------

            scale_tensor = torch.tensor(
                [
                    original_w,
                    original_h,
                    original_w,
                    original_h,
                ],
                dtype=torch.float32,
            )

            for i in range(
                detections.size(1)
            ):

                j = 0

                while (
                    j
                    < detections.size(2)
                ):

                    confidence = float(
                        detections[
                            0,
                            i,
                            j,
                            0,
                        ]
                    )

                    if confidence <= conf_th:
                        break

                    points = (
                        detections[
                            0,
                            i,
                            j,
                            1:5,
                        ]
                        .cpu()
                        * scale_tensor
                    ).numpy()

                    # NOTE: no "points /= scale_factor" here —
                    # that extra division was the bug.

                    x1, y1, x2, y2 = (
                        points
                    )

                    x1 = max(
                        0.0,
                        min(
                            float(
                                original_w
                            ),
                            float(x1),
                        ),
                    )

                    y1 = max(
                        0.0,
                        min(
                            float(
                                original_h
                            ),
                            float(y1),
                        ),
                    )

                    x2 = max(
                        0.0,
                        min(
                            float(
                                original_w
                            ),
                            float(x2),
                        ),
                    )

                    y2 = max(
                        0.0,
                        min(
                            float(
                                original_h
                            ),
                            float(y2),
                        ),
                    )

                    box_w = x2 - x1
                    box_h = y2 - y1

                    if (
                        box_w <= 0
                        or
                        box_h <= 0
                    ):

                        j += 1
                        continue

                    # --------------------------------------------
                    # Sanity filter #1: reject boxes that don't
                    # look like a human face by shape. A real
                    # face crop is roughly square-ish, never
                    # extremely thin or extremely wide.
                    # --------------------------------------------

                    aspect_ratio = box_w / box_h

                    if not (
                        0.55 <= aspect_ratio <= 1.8
                    ):

                        if (
                            DEBUG
                            and self._debug_print_count
                            < self._debug_print_limit
                        ):

                            print(
                                "[S3FD-debug] rejected: "
                                "aspect_ratio="
                                f"{aspect_ratio:.2f} "
                                f"conf={confidence:.2f} "
                                f"size={box_w:.0f}x{box_h:.0f}"
                            )

                            self._debug_print_count += 1

                        j += 1
                        continue

                    # --------------------------------------------
                    # Sanity filter #2: minimum size. Small
                    # patches (fingers, paper corners) that
                    # happen to pass the aspect-ratio check are
                    # rejected here.
                    # --------------------------------------------

                    if (
                        box_w < MIN_FACE_SIZE
                        or
                        box_h < MIN_FACE_SIZE
                    ):

                        if (
                            DEBUG
                            and self._debug_print_count
                            < self._debug_print_limit
                        ):

                            print(
                                "[S3FD-debug] rejected: "
                                "too small "
                                f"size={box_w:.0f}x{box_h:.0f} "
                                f"(min={MIN_FACE_SIZE}) "
                                f"conf={confidence:.2f}"
                            )

                            self._debug_print_count += 1

                        j += 1
                        continue

                    # --------------------------------------------
                    # Sanity filter #3 (NEW): Haar cascade
                    # cross-check. This is the independent,
                    # structural verification — it catches the
                    # cases S3FD genuinely scores as high
                    # confidence (so filters #1/#2/conf_th alone
                    # can't reject them), such as pillow prints
                    # and blanket patterns.
                    # --------------------------------------------

                    if self.haar_cascade is not None:

                        ix1, iy1 = (
                            int(x1),
                            int(y1),
                        )

                        ix2, iy2 = (
                            int(x2),
                            int(y2),
                        )

                        # Small padding so Haar has a bit of
                        # context around a tight S3FD box.
                        pad_x = int(
                            box_w * 0.15
                        )

                        pad_y = int(
                            box_h * 0.15
                        )

                        crop_x1 = max(
                            0,
                            ix1 - pad_x,
                        )

                        crop_y1 = max(
                            0,
                            iy1 - pad_y,
                        )

                        crop_x2 = min(
                            original_w,
                            ix2 + pad_x,
                        )

                        crop_y2 = min(
                            original_h,
                            iy2 + pad_y,
                        )

                        face_patch = gray_full[
                            crop_y1:crop_y2,
                            crop_x1:crop_x2,
                        ]

                        if face_patch.size == 0:

                            j += 1
                            continue

                        # Histogram equalization: boosts local
                        # contrast so Haar can still find eye/
                        # nose/mouth structure in dim or
                        # unevenly lit rooms (webcams in low
                        # light produce flat, low-contrast
                        # frames that the raw grayscale crop
                        # often fails on).
                        face_patch = cv2.equalizeHist(
                            face_patch
                        )

                        haar_matches = (
                            self.haar_cascade
                            .detectMultiScale(
                                face_patch,
                                scaleFactor=1.05,
                                minNeighbors=3,
                                minSize=(20, 20),
                            )
                        )

                        if len(haar_matches) == 0:

                            if (
                                DEBUG
                                and self._debug_print_count
                                < self._debug_print_limit
                            ):

                                print(
                                    "[S3FD-debug] rejected: "
                                    "Haar found no face "
                                    "structure in box "
                                    f"conf={confidence:.2f} "
                                    "size="
                                    f"{box_w:.0f}x{box_h:.0f}"
                                )

                                self._debug_print_count += 1

                            # S3FD says "face" here but no
                            # eye/nose/mouth structure was
                            # found — reject it.
                            j += 1
                            continue

                    if (
                        DEBUG
                        and self._debug_print_count
                        < self._debug_print_limit
                    ):

                        print(
                            "[S3FD-debug] ACCEPTED box "
                            f"conf={confidence:.2f} "
                            f"size={box_w:.0f}x{box_h:.0f}"
                        )

                        self._debug_print_count += 1

                    all_boxes.append(
                        [
                            x1,
                            y1,
                            x2,
                            y2,
                            confidence,
                        ]
                    )

                    j += 1

        if not all_boxes:

            return np.empty(
                (0, 5),
                dtype=np.float32,
            )

        boxes = np.asarray(
            all_boxes,
            dtype=np.float32,
        )

        keep = nms_(
            boxes,
            0.10,
        )

        return boxes[
            keep
        ]


# ============================================================
# TALKNET ASD
# ============================================================

class TalkNetASD(nn.Module):

    def __init__(
        self,
        weights_path=TALKNET_MODEL_PATH,
        device=TORCH_DEVICE,
    ):

        super().__init__()

        self.device = torch.device(
            device
        )

        print(
            "[TalkNet] Initializing "
            f"TalkNet on {self.device}"
        )

        self.model = (
            talkNetModel()
            .to(self.device)
        )

        self.lossAV = (
            lossAV()
            .to(self.device)
        )

        self.load_parameters(
            weights_path
        )

        self.model.eval()
        self.lossAV.eval()

        print(
            "[TalkNet] Model loaded "
            "successfully."
        )

    # ========================================================
    # LOAD TALKNET WEIGHTS
    # ========================================================

    def load_parameters(
        self,
        path,
    ):

        path = str(path)

        if not os.path.isfile(
            path
        ):

            raise FileNotFoundError(
                "\n"
                "[TalkNet] Missing "
                "TalkNet weights:\n"
                f"{path}\n\n"
                "Expected file:\n"
                "weights/pretrain_TalkSet.model"
            )

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        if (
            isinstance(
                checkpoint,
                dict,
            )
            and
            "state_dict" in checkpoint
        ):

            checkpoint = (
                checkpoint[
                    "state_dict"
                ]
            )

        if not isinstance(
            checkpoint,
            dict,
        ):

            raise RuntimeError(
                "[TalkNet] Invalid "
                "checkpoint format."
            )

        own_state = self.state_dict()

        loaded_count = 0

        skipped = []

        for name, parameter in (
            checkpoint.items()
        ):

            clean_name = name.replace(
                "module.",
                "",
            )

            if clean_name not in own_state:

                skipped.append(
                    f"{name}: missing"
                )

                continue

            parameter = (
                parameter.to(
                    self.device
                )
            )

            if (
                own_state[
                    clean_name
                ].shape
                != parameter.shape
            ):

                skipped.append(
                    f"{name}: shape mismatch"
                )

                continue

            own_state[
                clean_name
            ].copy_(
                parameter
            )

            loaded_count += 1

        expected_count = len(
            own_state
        )

        if loaded_count != expected_count:

            raise RuntimeError(
                "[TalkNet] Checkpoint "
                "does not match the model.\n"
                f"Loaded "
                f"{loaded_count}/"
                f"{expected_count} parameters.\n"
                + "\n".join(
                    skipped[:20]
                )
            )

        print(
            "[TalkNet] Loaded "
            f"{loaded_count}/"
            f"{expected_count} parameters."
        )

    # ========================================================
    # TALKING SCORE
    # ========================================================

    @torch.inference_mode()
    def compute_talking_scores(
        self,
        audio_data,
        sample_rate,
        face_crops,
    ):

        if (
            audio_data is None
            or
            len(audio_data) == 0
        ):

            return -10.0, []

        if (
            face_crops is None
            or
            len(face_crops) < 5
        ):

            return -10.0, []

        # ====================================================
        # AUDIO
        # ====================================================

        audio = np.asarray(
            audio_data,
            dtype=np.float32,
        ).reshape(
            -1
        )

        if audio.size == 0:

            return -10.0, []

        max_audio = float(
            np.max(
                np.abs(audio)
            )
        )

        # Handle int16-style microphone data
        if max_audio > 2.0:

            audio /= 32768.0

        # ====================================================
        # RESAMPLE
        # ====================================================

        if sample_rate != AUDIO_SAMPLE_RATE:

            # NumPy-only linear resampling.
            # Avoids SciPy/NumPy compatibility problems.
            new_length = int(
                len(audio)
                * AUDIO_SAMPLE_RATE
                / sample_rate
            )

            if new_length <= 0:

                return -10.0, []

            old_positions = np.linspace(
                0.0,
                1.0,
                num=len(audio),
                endpoint=False,
                dtype=np.float64,
            )

            new_positions = np.linspace(
                0.0,
                1.0,
                num=new_length,
                endpoint=False,
                dtype=np.float64,
            )

            audio = np.interp(
                new_positions,
                old_positions,
                audio,
            ).astype(
                np.float32
            )

            sample_rate = (
                AUDIO_SAMPLE_RATE
            )

        # ====================================================
        # MFCC
        # ====================================================

        audio_feature = (
            python_speech_features.mfcc(
                audio,
                sample_rate,
                numcep=13,
                winlen=0.025,
                winstep=0.010,
            )
        ).astype(
            np.float32
        )

        # ====================================================
        # VISUAL / FACE CROPS
        # ====================================================

        visual_features = []

        for face_img in face_crops:

            if face_img is None:

                continue

            if face_img.ndim == 3:

                face_img = cv2.cvtColor(
                    face_img,
                    cv2.COLOR_BGR2GRAY,
                )

            face_img = cv2.resize(
                face_img,
                (
                    TALKNET_FACE_SIZE,
                    TALKNET_FACE_SIZE,
                ),
                interpolation=cv2.INTER_AREA,
            )

            visual_features.append(
                face_img
            )

        if len(
            visual_features
        ) < 5:

            return -10.0, []

        visual_features = np.asarray(
            visual_features,
            dtype=np.float32,
        )

        # ====================================================
        # TALKNET TEMPORAL WINDOW
        # ====================================================

        num_video_frames = (
            visual_features.shape[0]
        )

        # TalkNet uses:
        #
        # 25 video frames
        # =
        # 100 audio/MFCC frames
        #
        # Therefore:
        #
        # audio frames = video frames * 4

        required_audio_frames = (
            num_video_frames * 4
        )

        # ====================================================
        # AUDIO LENGTH FIX
        # ====================================================

        if (
            audio_feature.shape[0]
            < required_audio_frames
        ):

            shortage = (
                required_audio_frames
                -
                audio_feature.shape[0]
            )

            if audio_feature.shape[0] > 0:

                audio_feature = np.pad(
                    audio_feature,
                    (
                        (0, shortage),
                        (0, 0),
                    ),
                    mode="wrap",
                )

            else:

                return -10.0, []

        audio_feature = (
            audio_feature[
                :required_audio_frames
            ]
        )

        # ====================================================
        # FINAL SHAPE CHECKS
        # ====================================================

        if audio_feature.shape != (
            required_audio_frames,
            13,
        ):

            raise RuntimeError(
                "[TalkNet] Invalid audio "
                "feature shape: "
                f"{audio_feature.shape}; "
                "expected "
                f"({required_audio_frames}, 13)"
            )

        if visual_features.shape != (
            num_video_frames,
            TALKNET_FACE_SIZE,
            TALKNET_FACE_SIZE,
        ):

            raise RuntimeError(
                "[TalkNet] Invalid visual "
                "feature shape: "
                f"{visual_features.shape}; "
                "expected "
                f"({num_video_frames}, "
                f"{TALKNET_FACE_SIZE}, "
                f"{TALKNET_FACE_SIZE})"
            )

        # ====================================================
        # TORCH INPUT
        # ====================================================

        in_audio = (
            torch.from_numpy(
                audio_feature
            )
            .unsqueeze(
                0
            )
            .to(
                self.device
            )
        )

        in_visual = (
            torch.from_numpy(
                visual_features
            )
            .unsqueeze(
                0
            )
            .to(
                self.device
            )
        )

        # ====================================================
        # AUDIO FRONTEND
        # ====================================================

        audio_embed = (
            self.model
            .forward_audio_frontend(
                in_audio
            )
        )

        # ====================================================
        # VISUAL FRONTEND
        # ====================================================

        visual_embed = (
            self.model
            .forward_visual_frontend(
                in_visual
            )
        )

        # ====================================================
        # CROSS ATTENTION
        # ====================================================

        (
            audio_embed,
            visual_embed,
        ) = (
            self.model
            .forward_cross_attention(
                audio_embed,
                visual_embed,
            )
        )

        # ====================================================
        # AUDIO-VISUAL BACKEND
        # ====================================================

        logits = (
            self.model
            .forward_audio_visual_backend(
                audio_embed,
                visual_embed,
            )
        )

        # ====================================================
        # TALKNET CLASSIFIER
        # ====================================================

        scores = self.lossAV.forward(
            logits,
            labels=None,
        )

        scores = np.asarray(
            scores,
            dtype=np.float32,
        ).reshape(
            -1
        )

        if scores.size == 0:

            return -10.0, []

        # ====================================================
        # FINAL SCORE
        # ====================================================

        return (
            float(
                np.mean(scores)
            ),
            scores.tolist(),
        )