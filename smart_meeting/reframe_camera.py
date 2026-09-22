# ============================================================
# REFRAME-DRIVEN SPEAKER CAMERA
# ------------------------------------------------------------
# Replaces two things that used to live in speaker_engine.py:
#   1) the ad-hoc speaker_switch_counter / speaker_release_counter
#      hysteresis that decided `active_speaker_id`
#   2) the ad-hoc cam_cx / cam_cy / cam_half_w EMA smoothing that
#      decided where the crop window sat
#
# Both are now reframe's job (pip package "reframe", core only,
# pure python, zero deps: https://github.com/vxnuaj/reframe):
#   - SubjectSelector: locks onto a subject and only switches when a
#     challenger clearly out-talks it for `turn_hold_frames` in a row
#     (stops the camera ping-ponging on every backchannel "mm-hmm").
#   - Camera: a damped-spring follower for center + zoom (stops the
#     snap/jitter the old EMA-of-a-single-alpha approach had).
#   - reassoc_radius also quietly absorbs *some* of the tracker-id
#     churn in PersonTrack (a dropped/re-numbered box near where the
#     locked subject just was is treated as the same subject for
#     camera purposes) -- it does NOT fix Person-label churn itself,
#     that still needs face-embedding re-ID (separate, not done here).
#
# Our own S3FD + TalkNet stay exactly as they are -- this module only
# repackages PersonTrack objects into reframe.types.Detection each
# frame and asks reframe who to lock onto and where the camera goes.
# ============================================================

import math

from reframe.rank import SubjectSelector
from reframe.smooth import Camera, base_crop_size, clamp
from reframe.presets import resolve_preset
from reframe.types import Detection

from smart_meeting.config import (
    TARGET_WIDTH,
    TARGET_HEIGHT,
    TALK_THRESHOLD,
)


def talknet_score_to_speaker_score(raw_score):
    """
    reframe's speaker_score is expected in 0..1 (a "how clearly is this
    person talking right now" probability). Our TalkNet score is a raw
    logit compared against TALK_THRESHOLD (config.py, default -1.0), not
    a probability. Center a sigmoid on TALK_THRESHOLD so:
        raw_score == TALK_THRESHOLD  -> 0.5
        raw_score  > TALK_THRESHOLD  -> > 0.5 (increasingly confident talker)
        raw_score  < TALK_THRESHOLD  -> < 0.5 (increasingly clearly silent)
    If TalkNet's real logit spread on your footage is much wider/narrower
    than ~1 unit, tune SCORE_SPREAD below rather than TALK_THRESHOLD --
    that keeps the "is this person talking at all" cutoff where it was.
    """
    SCORE_SPREAD = 1.0
    x = (raw_score - TALK_THRESHOLD) / SCORE_SPREAD
    x = clamp(x, -20.0, 20.0)  # avoid overflow in exp() on wild scores
    return 1.0 / (1.0 + math.exp(-x))


class ReframeCamera:
    """
    One instance lives on SmartMeetingDirector. Call update(tracks) once
    per processed frame; it returns the locked subject id and the crop
    window (cx, cy, crop_w, crop_h) the frame should be cropped to.
    """

    def __init__(self, frame_w, frame_h, preset_overrides=None):
        self.frame_w = frame_w
        self.frame_h = frame_h

        # talking_head is reframe's only tuned preset and it matches this
        # shot exactly: a locked head-and-shoulders view with turn-taking.
        # override zoom range so it doesn't punch in tighter than our
        # face crops can comfortably support.
        overrides = {"min_zoom": 1.0, "max_zoom": 1.6}
        overrides.update(preset_overrides or {})
        self.preset = resolve_preset("talking_head", overrides)

        self.selector = SubjectSelector(
            min_hold_frames=18,
            switch_margin=0.8,
            turn_hold_frames=24,        # ~2.4s at TARGET_FPS=10 before a
                                         # challenger can take the lock
            speaker_floor=0.55,         # our sigmoid centers "is talking"
                                         # at 0.5; 0.55 = clearly above the
                                         # TALK_THRESHOLD cutoff
            speaker_switch_margin=0.05,
            reassoc_radius=0.18,        # a little wider than reframe's
                                         # default -- our webcam boxes move
                                         # more than edited talking-head footage
        )

        aspect = (TARGET_WIDTH, TARGET_HEIGHT)
        self.camera = Camera(
            frame_w,
            frame_h,
            aspect,
            self.preset,
            init_cx=frame_w / 2.0,
            init_cy=frame_h / 2.0,
            init_zoom=self.preset.min_zoom,
        )
        self.base_w, self.base_h = base_crop_size(
            frame_w, frame_h, aspect[0], aspect[1]
        )

        self.active_speaker_id = None

    # --------------------------------------------------------
    def _tracks_to_detections(self, tracks):
        dets = []
        for t in tracks:
            x1, y1, x2, y2 = t.smooth_box
            cx, cy = t.center
            dets.append(
                Detection(
                    cls_name="person",
                    conf=1.0,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    track_id=t.id,
                    has_face=True,
                    face_cx=cx,
                    face_cy=cy,
                    speaker_score=talknet_score_to_speaker_score(t.last_score),
                )
            )
        return dets

    # --------------------------------------------------------
    def update(self, tracks):
        """
        tracks: current list of PersonTrack (a snapshot taken under
        SmartMeetingDirector's lock, same as the old code did).
        Returns (active_speaker_id, cx, cy, crop_w, crop_h).
        """
        dets = self._tracks_to_detections(tracks)
        subject = self.selector.select(dets, self.frame_w, self.frame_h)

        if subject is not None:
            fx, fy = subject.face_cx, subject.face_cy
            desired_zoom = clamp(
                (0.62 * self.base_h) / max(subject.h, 1.0),
                self.preset.min_zoom,
                self.preset.max_zoom,
            )
            conf = clamp(subject.conf, 0.2, 1.0)
            self.active_speaker_id = subject.track_id
        else:
            # graceful hold, same idea as reframe's build_crop_path: keep the
            # last framing instead of snapping to frame center.
            fx, fy = self.camera.x.target, self.camera.y.target
            desired_zoom = self.camera.zoom.target
            conf = 0.15
            # do NOT clear active_speaker_id here -- max_missed_frames inside
            # SubjectSelector already governs when a dropout means "gone for
            # good" vs a brief blip; let select() clear the lock itself.
            if self.selector.locked_track_id is None:
                self.active_speaker_id = None

        if self.selector.last_switch:
            self.camera.boost(6, factor=2.5)  # quick eased whip to the new speaker

        cx, cy, z = self.camera.update(fx, fy, desired_zoom, conf)
        cw, ch = self.camera.crop_size(z)
        return self.active_speaker_id, cx, cy, cw, ch

    # --------------------------------------------------------
    def crop_box(self, cx, cy, cw, ch):
        """cx, cy, cw, ch (as returned by update()) -> integer (x1, y1, x2, y2)
        clamped fully inside the frame."""
        x1 = int(round(clamp(cx - cw / 2.0, 0, self.frame_w - cw)))
        y1 = int(round(clamp(cy - ch / 2.0, 0, self.frame_h - ch)))
        x2 = min(self.frame_w, x1 + int(round(cw)))
        y2 = min(self.frame_h, y1 + int(round(ch)))
        return x1, y1, x2, y2