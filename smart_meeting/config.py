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

FACE_DET_CONF = 0.75
FACE_DET_SCALE = 1.0
FACE_DET_EVERY_N_FRAMES = 6

IOU_TRACK_THRESH = 0.25
MAX_FAILED_DET = 12
MIN_FACE_SIZE = 90


# ============================================================
# TALKNET
# ============================================================

TALK_THRESHOLD = 0.0
TALKNET_FACE_SIZE = 112

WINDOW_FRAMES = 25
WINDOW_SECONDS = 1.25


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

# Audio thresholds — raised significantly.
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
