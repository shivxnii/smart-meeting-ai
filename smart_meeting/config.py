# ============================================================
# SMART MEETING CONFIGURATION
# Hindi + English multilingual subtitle configuration
# ============================================================

import os
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

WEIGHTS_DIR = os.path.join(
    BASE_DIR,
    "weights"
)

TALKNET_MODEL_PATH = os.path.join(
    WEIGHTS_DIR,
    "pretrain_TalkSet.model"
)

S3FD_MODEL_PATH = os.path.join(
    WEIGHTS_DIR,
    "sfd_face.pth"
)

S3FD_WEIGHT_PATH = S3FD_MODEL_PATH


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CAMERA
# ============================================================

CAMERA_INDEX = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

TARGET_WIDTH = 640
TARGET_HEIGHT = 480

CAMERA_FPS = 10
TARGET_FPS = 10


# ============================================================
# FACE DETECTION
# ============================================================

FACE_DET_CONF = 0.5
FACE_DET_SCALE = 1.0
# Frames are now processed once per camera frame, so 3 is about the
# same wall-clock spacing as the old value of 6.
FACE_DET_EVERY_N_FRAMES = 3

IOU_TRACK_THRESH = 0.25
MAX_FAILED_DET = 12
MIN_FACE_SIZE = 90


# ============================================================
# TALKNET
# ============================================================

TALK_THRESHOLD = -1.0
TALKNET_FACE_SIZE = 112

WINDOW_FRAMES = 25
WINDOW_SECONDS = 1.25


# ============================================================
# AUDIO / VIDEO ALIGNMENT FOR TALKNET
# ============================================================

# Resample face crops to a regular 25 fps grid and feed TalkNet only the
# audio from the same 1 second. Set to False to get the old behaviour.
ALIGN_AV_WINDOW = True
AV_FPS = 25
AV_WINDOW_FRAMES = 25

# A grid slot counts as "covered" if a real frame is this close.
# Loosened from 0.30 / 0.70 — on CPU (S3FD + TalkNet + Whisper sharing
# one core) the real processed-frame rate is choppier than a clean 25fps
# grid, so the stricter values were rejecting every window
# (faces_available stuck at 0) and TalkNet's real lip-sync scoring never
# got to run at all.
AV_MAX_GAP_SEC = 0.50
AV_MIN_COVERAGE = 0.45

# Extra shift (seconds) removed from the END of the audio window.
# Webcams usually lag the microphone; try 0.1 to 0.3 if lips and voice
# look out of step.
AV_AUDIO_TRIM_SEC = 0.0

# Print per-track debug lines from the speaker engine.
DEBUG_TALKNET = True


# ============================================================
# MATCH THE ORIGINAL TALKNET INPUTS
# ============================================================

# Crop the face exactly like the original TalkNet demo (tight, mouth-centred
# 112x112). The old crop showed the whole head and lips looked too small.
TALKNET_REFERENCE_CROP = True

# TalkNet was trained on 16-bit audio values (about +-32768). Microphone
# audio arrives as +-1.0, which shifts the first MFCC value by about 20.8.
# True multiplies by 32768 before computing MFCC to match training.
TALKNET_INT16_AUDIO_SCALE = True

# If exactly one person is in view and there is clear voice activity, credit
# that person even when TalkNet is unsure. Avoids "Unknown Speaker" in
# one-to-one calls. Has no effect with two or more faces in view.
SINGLE_PERSON_FALLBACK = True


# ============================================================
# ACTIVE SPEAKER STABILITY
# ============================================================

SPEAKER_SWITCH_HYSTERESIS = 3
SPEAKER_RELEASE_HYSTERESIS = 5


# ============================================================
# CAMERA FRAMING
# ============================================================

ASPECT_RATIO = TARGET_WIDTH / TARGET_HEIGHT

FRAMING_PADDING_FACTOR = 2.8
FRAMING_SMOOTH_ALPHA = 0.20


# ============================================================
# AUDIO
# ============================================================

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_BLOCKSIZE = 1024

# Which microphone to use. None = sounddevice's default input device.
# If the app isn't picking up your voice, run this to list devices and
# find your mic's index number:
#   python -c "import sounddevice as sd; print(sd.query_devices())"
# then set AUDIO_DEVICE to that number, e.g. AUDIO_DEVICE = 1
AUDIO_DEVICE = None

# 3 seconds gives Whisper more context for complete sentences.
WINDOW_AUDIO_SAMPLES = 48000


# ============================================================
# WHISPER
# ============================================================

# "base" is more accurate than "tiny".
# If CPU is very slow, change this to "small" only if GPU is available.
WHISPER_MODEL = "base"

# Whisper should not run too frequently on CPU.
WHISPER_INTERVAL = 4.0

WHISPER_MIN_AUDIO_SECONDS = 1.5

# Automatic Hindi + English detection.
WHISPER_LANGUAGE = None

# Translate everything (Hindi/English/Hinglish) into English subtitles.
WHISPER_TASK = "translate"

# Audio thresholds â€” raised significantly.
# Old values (0.006 / 0.012) were letting background noise,
# fan hum, and quiet room sound through, which caused Whisper
# to hallucinate sentences on silence.
WHISPER_AUDIO_RMS_THRESHOLD = 0.008
WHISPER_AUDIO_PEAK_THRESHOLD = 0.05


# ============================================================
# PROCESSING
# ============================================================

MAX_FRAME_QUEUE = 2


# ============================================================
# SERVER
# ============================================================

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5000


# ============================================================
# DEBUG
# ============================================================

DEBUG = True


# ============================================================
# VALIDATION
# ============================================================

def validate_config():
    errors = []

    if not os.path.exists(TALKNET_MODEL_PATH):
        errors.append(
            "TalkNet model not found: "
            + TALKNET_MODEL_PATH
        )

    if not os.path.exists(S3FD_MODEL_PATH):
        errors.append(
            "S3FD model not found: "
            + S3FD_MODEL_PATH
        )

    if errors:
        print("\n[CONFIG] WARNING:")

        for error in errors:
            print("  - " + error)

        print()

        return False

    return True


print(
    "[CONFIG] Loaded multilingual subtitle configuration "
    "(Hindi + English, Whisper auto-detection)"
)

validate_config()

# 0 = unlimited (normal multi-person meeting, needed so the camera can
# switch focus between different speakers). Only set this to 1 if you are
# testing alone in front of the camera and want a hard guarantee that you
# can never be assigned a second ID.
MAX_PEOPLE = 0

# A visible track counts as "briefly lost" (and gets first claim on any
# new, unmatched face nearby) after this many missed detection frames.
STALE_MIN_MISSED = 3