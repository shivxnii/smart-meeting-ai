# ============================================================
# SMART MEETING - SPEAKER ENGINE
# ============================================================

import time
import threading
from collections import deque

import cv2
import numpy as np

from smart_meeting.config import (
    ASPECT_RATIO,
    FRAMING_PADDING_FACTOR,
    FRAMING_SMOOTH_ALPHA,
    TARGET_WIDTH,
    TARGET_HEIGHT,
    TALK_THRESHOLD,
    FACE_DET_CONF,
    FACE_DET_SCALE,
    FACE_DET_EVERY_N_FRAMES,
    IOU_TRACK_THRESH,
    MAX_FAILED_DET,
    WINDOW_FRAMES,
    WINDOW_SECONDS,
    SPEAKER_SWITCH_HYSTERESIS,
    SPEAKER_RELEASE_HYSTERESIS,
    TALKNET_FACE_SIZE,
    MIN_FACE_SIZE,
)

from smart_meeting.talknet_cpu import (
    TalkNetASD,
    S3FDDetector,
)

from smart_meeting.subtitle_engine import (
    SubtitleEngine,
)


# ============================================================
# IOU
# ============================================================

def compute_iou(box_a, box_b):

    x1 = max(
        float(box_a[0]),
        float(box_b[0]),
    )

    y1 = max(
        float(box_a[1]),
        float(box_b[1]),
    )

    x2 = min(
        float(box_a[2]),
        float(box_b[2]),
    )

    y2 = min(
        float(box_a[3]),
        float(box_b[3]),
    )

    intersection_width = max(
        0.0,
        x2 - x1,
    )

    intersection_height = max(
        0.0,
        y2 - y1,
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    if intersection <= 0:
        return 0.0

    area_a = max(
        1.0,
        float(box_a[2]) - float(box_a[0]),
    ) * max(
        1.0,
        float(box_a[3]) - float(box_a[1]),
    )

    area_b = max(
        1.0,
        float(box_b[2]) - float(box_b[0]),
    ) * max(
        1.0,
        float(box_b[3]) - float(box_b[1]),
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# CENTER DISTANCE
# ============================================================

def center_distance(box_a, box_b):

    ax = (
        float(box_a[0])
        + float(box_a[2])
    ) / 2.0

    ay = (
        float(box_a[1])
        + float(box_a[3])
    ) / 2.0

    bx = (
        float(box_b[0])
        + float(box_b[2])
    ) / 2.0

    by = (
        float(box_b[1])
        + float(box_b[3])
    ) / 2.0

    return float(
        np.sqrt(
            (ax - bx) ** 2
            +
            (ay - by) ** 2
        )
    )


# ============================================================
# PERSON TRACK
# ============================================================

class PersonTrack:

    _next_id = 1

    def __init__(self, box):

        self.id = PersonTrack._next_id
        PersonTrack._next_id += 1

        self.box = list(box)

        self.smooth_box = list(box)

        self.missed_frames = 0

        self.age = 1

        self.visible_frames = 1

        self.face_history = deque(
            maxlen=max(
                WINDOW_FRAMES * 3,
                60,
            )
        )

        self.last_score = -10.0

        self.is_speaking = False

        self.last_seen = time.time()

    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    def update(
        self,
        box,
        face_crop,
        timestamp,
    ):

        self.missed_frames = 0

        self.age += 1

        self.visible_frames += 1

        self.last_seen = timestamp

        # Smoother than directly replacing the
        # bounding box every frame.
        alpha = 0.45

        for i in range(4):

            self.smooth_box[i] = (
                (
                    1.0 - alpha
                )
                * self.smooth_box[i]
                +
                alpha * float(box[i])
            )

        self.box = list(box)

        if face_crop is not None:

            self.face_history.append(
                (
                    timestamp,
                    face_crop,
                )
            )

    # --------------------------------------------------------
    # MISSED
    # --------------------------------------------------------

    def mark_missed(self):

        self.missed_frames += 1

    # --------------------------------------------------------
    # CENTER
    # --------------------------------------------------------

    @property
    def center(self):

        return (
            (
                self.smooth_box[0]
                +
                self.smooth_box[2]
            ) / 2.0,

            (
                self.smooth_box[1]
                +
                self.smooth_box[3]
            ) / 2.0,
        )

    # --------------------------------------------------------
    # SIZE
    # --------------------------------------------------------

    @property
    def size(self):

        width = max(
            1.0,
            self.smooth_box[2]
            -
            self.smooth_box[0],
        )

        height = max(
            1.0,
            self.smooth_box[3]
            -
            self.smooth_box[1],
        )

        return max(
            width,
            height,
        )

    # --------------------------------------------------------
    # FACE WINDOW
    # --------------------------------------------------------

    def get_faces_for_window(
        self,
        end_time,
    ):

        start_time = (
            end_time
            - WINDOW_SECONDS
        )

        selected = []

        for timestamp, crop in (
            self.face_history
        ):

            if (
                start_time
                <= timestamp
                <= end_time
            ):

                selected.append(
                    crop
                )

        if len(selected) > WINDOW_FRAMES:

            selected = selected[
                -WINDOW_FRAMES:
            ]

        return selected


# ============================================================
# SMART MEETING DIRECTOR
# ============================================================

class SmartMeetingDirector:

    def __init__(self):

        print(
            "[Director] Loading AI models..."
        )

        self.face_detector = (
            S3FDDetector()
        )

        self.talknet = (
            TalkNetASD()
        )

        self.subtitle_engine = (
            SubtitleEngine(  model_name="base"  )
        )

        self.tracks = []

        self.active_speaker_id = None

        self.active_speaker_score = -10.0

        self.speaker_switch_counter = 0

        self.speaker_release_counter = 0

        self.frame_counter = 0

        self.last_detection_boxes = []

        self.last_detection_time = 0.0

        self.person_count = 0

        self.cam_cx = None
        self.cam_cy = None
        self.cam_half_w = None

        self.lock = threading.Lock()

        print(
            "[Director] Ready."
        )

    # ========================================================
    # FILTER DETECTIONS
    # ========================================================

    def _filter_detections(
        self,
        detections,
        frame_shape,
    ):

        height, width = frame_shape[:2]

        boxes = []

        for detection in detections:

            if len(detection) < 4:
                continue

            try:

                x1 = float(detection[0])
                y1 = float(detection[1])
                x2 = float(detection[2])
                y2 = float(detection[3])

                confidence = (
                    float(detection[4])
                    if len(detection) >= 5
                    else 1.0
                )

            except Exception:

                continue

            if confidence < FACE_DET_CONF:
                continue

            # Clamp.
            x1 = max(
                0.0,
                min(
                    float(width - 1),
                    x1,
                ),
            )

            y1 = max(
                0.0,
                min(
                    float(height - 1),
                    y1,
                ),
            )

            x2 = max(
                0.0,
                min(
                    float(width),
                    x2,
                ),
            )

            y2 = max(
                0.0,
                min(
                    float(height),
                    y2,
                ),
            )

            face_width = x2 - x1
            face_height = y2 - y1

            if (
                face_width < MIN_FACE_SIZE
                or
                face_height < MIN_FACE_SIZE
            ):
                continue

            boxes.append(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                ]
            )

        # ----------------------------------------------------
        # Remove duplicate/overlapping detections.
        # ----------------------------------------------------

        if len(boxes) <= 1:

            return boxes

        keep = []

        # Largest/highest quality boxes first.
        boxes.sort(
            key=lambda b: (
                (b[2] - b[0])
                *
                (b[3] - b[1])
            ),
            reverse=True,
        )

        for box in boxes:

            duplicate = False

            for existing in keep:

                iou = compute_iou(
                    box,
                    existing,
                )

                if iou >= 0.55:

                    duplicate = True
                    break

            if not duplicate:

                keep.append(box)

        return keep

    # ========================================================
    # FACE DETECTION
    # ========================================================

    def detect_faces_in_frame(
        self,
        frame_bgr,
    ):

        should_detect = (
            self.frame_counter
            % max(
                1,
                FACE_DET_EVERY_N_FRAMES,
            )
            == 0
        )

        # Reuse the previous detections between
        # S3FD passes.
        if (
            not should_detect
            and
            self.last_detection_boxes
        ):

            return list(
                self.last_detection_boxes
            )

        try:

            rgb = cv2.cvtColor(
                frame_bgr,
                cv2.COLOR_BGR2RGB,
            )

            detections = (
                self.face_detector
                .detect_faces(
                    rgb,
                    conf_th=FACE_DET_CONF,
                    scales=(
                        FACE_DET_SCALE,
                    ),
                )
            )

            boxes = self._filter_detections(
                detections,
                frame_bgr.shape,
            )

            self.last_detection_boxes = (
                boxes
            )

            self.last_detection_time = (
                time.time()
            )

            return list(boxes)

        except Exception as exc:

            print(
                "[Director] S3FD error:",
                repr(exc),
            )

            return []

    # ========================================================
    # FACE CROP
    # ========================================================

    def _extract_face_crop(
        self,
        frame_bgr,
        box,
    ):

        height, width = (
            frame_bgr.shape[:2]
        )

        x1, y1, x2, y2 = [
            int(round(v))
            for v in box
        ]

        x1 = max(
            0,
            min(
                width - 1,
                x1,
            ),
        )

        y1 = max(
            0,
            min(
                height - 1,
                y1,
            ),
        )

        x2 = max(
            x1 + 1,
            min(
                width,
                x2,
            ),
        )

        y2 = max(
            y1 + 1,
            min(
                height,
                y2,
            ),
        )

        face_width = x2 - x1
        face_height = y2 - y1

        size = max(
            face_width,
            face_height,
        )

        # More stable context around face.
        pad = int(
            size * 0.30
        )

        cx = (
            x1 + x2
        ) // 2

        cy = (
            y1 + y2
        ) // 2

        px1 = max(
            0,
            cx
            - size // 2
            - pad,
        )

        py1 = max(
            0,
            cy
            - size // 2
            - pad,
        )

        px2 = min(
            width,
            cx
            + size // 2
            + pad,
        )

        py2 = min(
            height,
            cy
            + size // 2
            + pad,
        )

        if (
            px2 <= px1
            or
            py2 <= py1
        ):

            return None

        crop = frame_bgr[
            py1:py2,
            px1:px2,
        ]

        if crop.size == 0:

            return None

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.resize(
            gray,
            (
                TALKNET_FACE_SIZE,
                TALKNET_FACE_SIZE,
            ),
            interpolation=cv2.INTER_AREA,
        )

        return gray

    # ========================================================
    # TRACK MATCHING
    # ========================================================

    def _match_detection_to_track(
        self,
        track,
        box,
    ):

        iou = compute_iou(
            track.smooth_box,
            box,
        )

        if iou >= IOU_TRACK_THRESH:

            return True

        # If IoU becomes temporarily small because
        # the face detector shifts slightly, allow
        # a center-distance based match.
        distance = center_distance(
            track.smooth_box,
            box,
        )

        track_size = max(
            1.0,
            track.size,
        )

        if (
            distance
            <= track_size * 0.85
        ):

            return True

        return False

    # ========================================================
    # TRACK UPDATE
    # ========================================================

    def update_tracks(
        self,
        frame_bgr,
        detected_boxes,
        timestamp,
    ):

        with self.lock:

            matched_tracks = set()

            matched_boxes = set()

            candidates = []

            # ------------------------------------------------
            # Build matching candidates.
            # ------------------------------------------------

            for track_index, track in enumerate(
                self.tracks
            ):

                for box_index, box in enumerate(
                    detected_boxes
                ):

                    iou = compute_iou(
                        track.smooth_box,
                        box,
                    )

                    distance = center_distance(
                        track.smooth_box,
                        box,
                    )

                    normalized_distance = (
                        distance
                        /
                        max(
                            track.size,
                            1.0,
                        )
                    )

                    # Strong IoU gets priority.
                    if iou >= IOU_TRACK_THRESH:

                        score = (
                            iou
                            +
                            0.15
                            *
                            max(
                                0.0,
                                1.0
                                -
                                normalized_distance,
                            )
                        )

                        candidates.append(
                            (
                                score,
                                track_index,
                                box_index,
                            )
                        )

                    # Loose center-distance recovery.
                    elif (
                        normalized_distance
                        <= 0.70
                    ):

                        score = (
                            0.35
                            -
                            normalized_distance
                            * 0.20
                        )

                        candidates.append(
                            (
                                score,
                                track_index,
                                box_index,
                            )
                        )

            candidates.sort(
                reverse=True
            )

            # ------------------------------------------------
            # Greedy assignment.
            # ------------------------------------------------

            for (
                score,
                track_index,
                box_index,
            ) in candidates:

                if (
                    track_index
                    in matched_tracks
                ):
                    continue

                if (
                    box_index
                    in matched_boxes
                ):
                    continue

                track = self.tracks[
                    track_index
                ]

                box = detected_boxes[
                    box_index
                ]

                if not self._match_detection_to_track(
                    track,
                    box,
                ):

                    continue

                crop = (
                    self._extract_face_crop(
                        frame_bgr,
                        box,
                    )
                )

                track.update(
                    box,
                    crop,
                    timestamp,
                )

                matched_tracks.add(
                    track_index
                )

                matched_boxes.add(
                    box_index
                )

            # ------------------------------------------------
            # Create tracks only for genuinely
            # unmatched detections.
            # ------------------------------------------------

            for box_index, box in enumerate(
                detected_boxes
            ):

                if (
                    box_index
                    in matched_boxes
                ):
                    continue

                # Extra duplicate protection.
                already_near_track = False

                for track in self.tracks:

                    if (
                        compute_iou(
                            track.smooth_box,
                            box,
                        )
                        >= 0.20
                    ):

                        already_near_track = True
                        break

                    distance = center_distance(
                        track.smooth_box,
                        box,
                    )

                    if (
                        distance
                        <= max(
                            track.size * 0.50,
                            30.0,
                        )
                    ):

                        already_near_track = True
                        break

                if already_near_track:

                    continue

                crop = (
                    self._extract_face_crop(
                        frame_bgr,
                        box,
                    )
                )

                track = PersonTrack(
                    box
                )

                track.update(
                    box,
                    crop,
                    timestamp,
                )

                self.tracks.append(
                    track
                )

            # ------------------------------------------------
            # Handle tracks not detected this pass.
            # ------------------------------------------------

            surviving = []

            for index, track in enumerate(
                self.tracks
            ):

                if (
                    index
                    not in matched_tracks
                ):

                    # A newly-created track may not have
                    # appeared in matched_tracks yet.
                    if track.visible_frames > 1:

                        track.mark_missed()

                if (
                    track.missed_frames
                    <= MAX_FAILED_DET
                ):

                    surviving.append(
                        track
                    )

            self.tracks = surviving

            # ------------------------------------------------
            # Clean invalid active speaker.
            # ------------------------------------------------

            valid_ids = {
                track.id
                for track in self.tracks
            }

            if (
                self.active_speaker_id
                not in valid_ids
            ):

                self.active_speaker_id = None

                self.active_speaker_score = -10.0

            self.person_count = len(
                self.tracks
            )

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(
        self,
        frame_bgr,
        timestamp=None,
    ):

        if timestamp is None:

            timestamp = time.time()

        self.frame_counter += 1

        boxes = (
            self.detect_faces_in_frame(
                frame_bgr
            )
        )

        self.update_tracks(
            frame_bgr,
            boxes,
            timestamp,
        )

    # ========================================================
    # ACTIVE SPEAKER
    # ========================================================

    def evaluate_active_speaker(
        self,
        audio_data,
        sample_rate=16000,
        window_end_time=None,
    ):

        if window_end_time is None:

            window_end_time = time.time()

        with self.lock:

            tracks_snapshot = list(
                self.tracks
            )

        if (
            not tracks_snapshot
            or
            audio_data is None
            or
            len(audio_data) == 0
        ):

            return (
                self.active_speaker_id,
                -10.0,
            )

        best_id = None

        best_score = -10.0

        # ----------------------------------------------------
        # Evaluate each tracked face.
        # ----------------------------------------------------

        for track in tracks_snapshot:

            faces = (
                track
                .get_faces_for_window(
                    window_end_time
                )
            )

            required_frames = max(
                10,
                int(
                    WINDOW_FRAMES
                    * 0.60
                ),
            )

            if (
                len(faces)
                < required_frames
            ):

                continue

            if (
                len(faces)
                > WINDOW_FRAMES
            ):

                indices = np.linspace(
                    0,
                    len(faces) - 1,
                    WINDOW_FRAMES,
                ).astype(int)

                faces = [
                    faces[index]
                    for index in indices
                ]

            try:

                score, _ = (
                    self.talknet
                    .compute_talking_scores(
                        audio_data,
                        sample_rate,
                        faces,
                    )
                )

            except Exception as exc:

                print(
                    "[TalkNet] Track "
                    f"{track.id} error:",
                    repr(exc),
                )

                continue

            track.last_score = float(
                score
            )

            if score > best_score:

                best_score = float(
                    score
                )

                best_id = track.id

        with self.lock:

            self.active_speaker_score = (
                best_score
            )

            # ------------------------------------------------
            # No candidate.
            # ------------------------------------------------

            if best_id is None:

                self.speaker_release_counter += 1

                self.speaker_switch_counter = 0

                if (
                    self.speaker_release_counter
                    >= SPEAKER_RELEASE_HYSTERESIS
                ):

                    self.active_speaker_id = None

                for track in self.tracks:

                    track.is_speaking = False

                return (
                    self.active_speaker_id,
                    best_score,
                )

            # ------------------------------------------------
            # No confident speech.
            # ------------------------------------------------

            if (
                best_score
                <= TALK_THRESHOLD
            ):

                self.speaker_release_counter += 1

                self.speaker_switch_counter = 0

                if (
                    self.speaker_release_counter
                    >= SPEAKER_RELEASE_HYSTERESIS
                ):

                    self.active_speaker_id = None

                for track in self.tracks:

                    track.is_speaking = False

                return (
                    self.active_speaker_id,
                    best_score,
                )

            # ------------------------------------------------
            # Confident speech.
            # ------------------------------------------------

            self.speaker_release_counter = 0

            if (
                best_id
                ==
                self.active_speaker_id
            ):

                self.speaker_switch_counter = 0

            else:

                self.speaker_switch_counter += 1

                if (
                    self.speaker_switch_counter
                    >= SPEAKER_SWITCH_HYSTERESIS
                ):

                    self.active_speaker_id = (
                        best_id
                    )

                    self.speaker_switch_counter = 0

            for track in self.tracks:

                track.is_speaking = (
                    track.id
                    ==
                    self.active_speaker_id
                )

            return (
                self.active_speaker_id,
                best_score,
            )

    # ========================================================
    # CENTERED SPEAKER VIEW
    # ========================================================

    def render_centered_speaker_frame(
        self,
        frame_bgr,
        subtitle_text="",
    ):

        height, width = (
            frame_bgr.shape[:2]
        )

        with self.lock:

            active_id = (
                self.active_speaker_id
            )

            active_track = None

            for track in self.tracks:

                if (
                    track.id
                    ==
                    active_id
                ):

                    active_track = track

                    break

        # ----------------------------------------------------
        # No active speaker.
        # ----------------------------------------------------

        if active_track is None:

            return (
                self.render_room_overview_frame(
                    frame_bgr,
                    subtitle_text,
                )
            )

        cx, cy = (
            active_track.center
        )

        person_size = (
            active_track.size
        )

        crop_height = max(
            120,
            int(
                person_size
                * FRAMING_PADDING_FACTOR
            ),
        )

        crop_width = int(
            crop_height
            * ASPECT_RATIO
        )

        crop_width = min(
            width,
            crop_width,
        )

        crop_height = min(
            height,
            crop_height,
        )

        target_cx = cx

        target_cy = cy

        target_half_width = (
            crop_width / 2.0
        )

        if self.cam_cx is None:

            self.cam_cx = target_cx

            self.cam_cy = target_cy

            self.cam_half_w = (
                target_half_width
            )

        else:

            alpha = (
                FRAMING_SMOOTH_ALPHA
            )

            self.cam_cx = (
                (1.0 - alpha)
                * self.cam_cx
                +
                alpha * target_cx
            )

            self.cam_cy = (
                (1.0 - alpha)
                * self.cam_cy
                +
                alpha * target_cy
            )

            self.cam_half_w = (
                (1.0 - alpha)
                * self.cam_half_w
                +
                alpha * target_half_width
            )

        crop_width = int(
            self.cam_half_w * 2.0
        )

        crop_width = max(
            100,
            min(
                width,
                crop_width,
            ),
        )

        crop_height = int(
            crop_width
            / ASPECT_RATIO
        )

        crop_height = max(
            100,
            min(
                height,
                crop_height,
            ),
        )

        x1 = int(
            self.cam_cx
            -
            crop_width / 2.0
        )

        y1 = int(
            self.cam_cy
            -
            crop_height / 2.0
        )

        x1 = max(
            0,
            min(
                x1,
                width - crop_width,
            ),
        )

        y1 = max(
            0,
            min(
                y1,
                height - crop_height,
            ),
        )

        x2 = min(
            width,
            x1 + crop_width,
        )

        y2 = min(
            height,
            y1 + crop_height,
        )

        crop = frame_bgr[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:

            return (
                self.render_room_overview_frame(
                    frame_bgr,
                    subtitle_text,
                )
            )

        # ------------------------------------------------
        # Hide/blur everyone except the active speaker
        # within this cropped region.
        # ------------------------------------------------

        with self.lock:
            other_tracks = [
                t for t in self.tracks
                if t.id != active_id
            ]

        for other_track in other_tracks:

            ox1, oy1, ox2, oy2 = [
                int(round(v)) for v in other_track.smooth_box
            ]

            cox1 = max(0, ox1 - x1)
            coy1 = max(0, oy1 - y1)
            cox2 = min(crop.shape[1], ox2 - x1)
            coy2 = min(crop.shape[0], oy2 - y1)

            if cox2 <= cox1 or coy2 <= coy1:
                continue

            region = crop[coy1:coy2, cox1:cox2]

            if region.size == 0:
                continue

            blurred_region = cv2.GaussianBlur(
                region,
                (51, 51),
                0,
            )

            crop[coy1:coy2, cox1:cox2] = blurred_region

        output = cv2.resize(
            crop,
            (
                TARGET_WIDTH,
                TARGET_HEIGHT,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        cv2.putText(
            output,
            (
                "ACTIVE SPEAKER: "
                f"Person {active_id}"
            ),
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 100),
            2,
            cv2.LINE_AA,
        )

        output = (
            SubtitleEngine
            .draw_subtitles_overlay(
                output,
                subtitle_text,
            )
        )

        return output

    # ========================================================
    # ROOM OVERVIEW
    # ========================================================

    def render_room_overview_frame(
        self,
        frame_bgr,
        subtitle_text="",
    ):

        display = frame_bgr.copy()

        with self.lock:

            tracks = list(
                self.tracks
            )

            active_id = (
                self.active_speaker_id
            )

        # ----------------------------------------------------
        # Draw tracked faces.
        # ----------------------------------------------------

        for track in tracks:

            x1, y1, x2, y2 = [
                int(round(v))
                for v in track.smooth_box
            ]

            x1 = max(
                0,
                min(
                    display.shape[1] - 1,
                    x1,
                ),
            )

            y1 = max(
                0,
                min(
                    display.shape[0] - 1,
                    y1,
                ),
            )

            x2 = max(
                x1 + 1,
                min(
                    display.shape[1],
                    x2,
                ),
            )

            y2 = max(
                y1 + 1,
                min(
                    display.shape[0],
                    y2,
                ),
            )

            active = (
                track.id
                ==
                active_id
                and
                track.is_speaking
            )

            if active:

                box_color = (
                    0,
                    255,
                    100,
                )

                thickness = 3

            else:

                box_color = (
                    255,
                    180,
                    0,
                )

                thickness = 2

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                box_color,
                thickness,
            )

            label = (
                f"Person {track.id}"
            )

            if active:

                label += (
                    "  SPEAKING "
                    f"{track.last_score:.2f}"
                )

            cv2.putText(
                display,
                label,
                (
                    x1,
                    max(
                        25,
                        y1 - 8,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                box_color,
                2,
                cv2.LINE_AA,
            )

        # ----------------------------------------------------
        # Room status.
        # ----------------------------------------------------

        cv2.putText(
            display,
            (
                "ROOM: "
                f"{len(tracks)} PERSON(S)"
            ),
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            (0, 210, 255),
            2,
            cv2.LINE_AA,
        )

        display = cv2.resize(
            display,
            (
                TARGET_WIDTH,
                TARGET_HEIGHT,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        return (
            SubtitleEngine
            .draw_subtitles_overlay(
                display,
                subtitle_text,
            )
        )

    # ========================================================
    # STATUS
    # ========================================================

    def get_status_summary(self):

        with self.lock:

            attendees = []

            for track in self.tracks:

                attendees.append(
                    {
                        "id": track.id,

                        "score": round(
                            float(
                                track.last_score
                            ),
                            3,
                        ),

                        "is_speaking":
                            bool(
                                track.is_speaking
                            ),

                        "missed_frames":
                            int(
                                track.missed_frames
                            ),
                    }
                )

            return {
                "persons_detected":
                    len(self.tracks),

                "active_speaker_id":
                    self.active_speaker_id,

                "speaking_score":
                    round(
                        float(
                            self.active_speaker_score
                        ),
                        3,
                    ),

                "is_speaking":
                    (
                        self.active_speaker_id
                        is not None
                    ),

                "attendees":
                    attendees,
            }
