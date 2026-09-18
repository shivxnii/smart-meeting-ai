# ============================================================
# SMART MEETING - MAIN APPLICATION
# ============================================================

import threading
import time

import cv2
import numpy as np
import sounddevice as sd

from flask import (
    Flask,
    Response,
    jsonify
)

from smart_meeting.config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    TARGET_FPS,
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    AUDIO_BLOCKSIZE,
    WINDOW_FRAMES,
    WINDOW_AUDIO_SAMPLES,
    WHISPER_INTERVAL,
    WHISPER_AUDIO_PEAK_THRESHOLD,
    SERVER_HOST,
    SERVER_PORT,
)

from smart_meeting.speaker_engine import (
    SmartMeetingDirector
)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# GLOBAL STATE
# ============================================================

frame_lock = threading.Lock()
audio_lock = threading.Lock()
state_lock = threading.Lock()

latest_frame = None
latest_output_frame = None
latest_subtitle = ""

latest_status = {
    "persons_detected": 0,
    "active_speaker_id": None,
    "speaking_score": 0.0,
    "is_speaking": False,
    "attendees": []
}


# ============================================================
# AUDIO BUFFER
# ============================================================

audio_buffer = []
MAX_AUDIO_BUFFER = WINDOW_AUDIO_SAMPLES * 4


# ============================================================
# VIDEO BUFFER
# ============================================================

video_buffer = []
MAX_VIDEO_BUFFER = WINDOW_FRAMES * 4


# ============================================================
# DIRECTOR
# ============================================================

director = None


# ============================================================
# AUDIO CALLBACK
# ============================================================

def audio_callback(indata, frames, time_info, status):
    if status:
        print("[Audio]", status)

    if indata is None:
        return

    try:
        samples = indata[:, 0].astype(np.float32).copy()

        with audio_lock:
            audio_buffer.extend(samples.tolist())

            if len(audio_buffer) > MAX_AUDIO_BUFFER:
                del audio_buffer[: len(audio_buffer) - MAX_AUDIO_BUFFER]

    except Exception as exc:
        print("[Audio] Callback error:", repr(exc))


# ============================================================
# CAMERA CAPTURE LOOP
# ============================================================

def camera_loop():
    global latest_frame

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("[Camera] ERROR: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    print("[Camera] Webcam started.")
    print(
        "[Camera] Resolution:",
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "x",
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("[Camera] Frame read failed.")
                time.sleep(0.03)
                continue

            timestamp = time.time()

            with frame_lock:
                latest_frame = frame

                video_buffer.append((timestamp, frame.copy()))

                if len(video_buffer) > MAX_VIDEO_BUFFER:
                    del video_buffer[: len(video_buffer) - MAX_VIDEO_BUFFER]

            time.sleep(0.001)

    finally:
        cap.release()
        print("[Camera] Webcam released.")


# ============================================================
# AI / FACE TRACKING LOOP
# ============================================================

def director_loop():
    print("[Director] AI processing thread started.")

    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.02)
            continue

        try:
            director.process_frame(frame, time.time())
        except Exception as exc:
            print("[Director] Frame error:", repr(exc))

        time.sleep(0.005)


# ============================================================
# VIDEO RENDER LOOP
# ============================================================

def render_loop():
    global latest_output_frame

    print("[Render] Video rendering thread started.")

    frame_interval = 1.0 / max(1, TARGET_FPS)
    next_render = time.perf_counter()

    while True:
        now = time.perf_counter()

        if now < next_render:
            time.sleep(min(0.005, next_render - now))
            continue

        next_render = now + frame_interval

        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            continue

        try:
            rendered = director.render_centered_speaker_frame(frame, "")

            with frame_lock:
                latest_output_frame = rendered.copy()

        except Exception as exc:
            print("[Render] Frame error:", repr(exc))


# ============================================================
# ANALYSIS LOOP
# ============================================================

def analysis_loop():
    global latest_subtitle
    global latest_status

    print("[Analysis] Waiting for enough audio/video...")

    last_whisper_time = 0.0

    while True:
        time.sleep(0.25)

        # --------------------------------------------------
        # AUDIO
        # --------------------------------------------------

        with audio_lock:
            if len(audio_buffer) < WINDOW_AUDIO_SAMPLES:
                continue

            audio = np.asarray(
                audio_buffer[-WINDOW_AUDIO_SAMPLES:],
                dtype=np.float32
            )

        # --------------------------------------------------
        # VIDEO
        # --------------------------------------------------

        with frame_lock:
            if latest_frame is None:
                continue

            current_frame = latest_frame.copy()

            if video_buffer:
                window_end_time = video_buffer[-1][0]
            else:
                window_end_time = time.time()

        # --------------------------------------------------
        # TALKNET
        # --------------------------------------------------

        active_id = None
        score = 0.0

        try:
            active_id, score = director.evaluate_active_speaker(
                audio,
                AUDIO_SAMPLE_RATE,
                window_end_time
            )
        except Exception as exc:
            print("[TalkNet] Analysis error:", repr(exc))

        # --------------------------------------------------
        # WHISPER
        # --------------------------------------------------

        subtitle = ""
        now = time.time()

        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio ** 2)))

        peak_ok = peak > WHISPER_AUDIO_PEAK_THRESHOLD
        interval_ok = (now - last_whisper_time) >= WHISPER_INTERVAL
        will_call_whisper = peak_ok and interval_ok

        print(
            "[DEBUG] peak={:.4f} rms={:.4f} peak_ok={} interval_ok={} will_call={}".format(
                peak, rms, peak_ok, interval_ok, will_call_whisper
            )
        )

        if peak_ok and interval_ok:
            if active_id is not None:
                speaker_name = f"Person {active_id}"
            else:
                speaker_name = "Unknown Speaker"

            try:
                subtitle = director.subtitle_engine.transcribe_audio_segment(
                    audio,
                    AUDIO_SAMPLE_RATE,
                    speaker_name
                )

                last_whisper_time = now

            except Exception as exc:
                print("[Whisper] Error:", repr(exc))

        # --------------------------------------------------
        # STATUS
        # --------------------------------------------------

        try:
            status = director.get_status_summary()
            status["speaking_score"] = round(float(score), 3)

            with state_lock:
                latest_status = status

                if subtitle:
                    latest_subtitle = subtitle

        except Exception as exc:
            print("[Status] Error:", repr(exc))


# ============================================================
# MJPEG STREAM
# ============================================================

def generate_frames():
    while True:
        with frame_lock:
            if latest_output_frame is not None:
                frame = latest_output_frame.copy()
            elif latest_frame is not None:
                frame = latest_frame.copy()
            else:
                frame = None

        if frame is None:
            time.sleep(0.05)
            continue

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )

        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + encoded.tobytes()
            + b"\r\n"
        )

        time.sleep(1.0 / max(1, TARGET_FPS))


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():
    return """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Smart Meeting — Active Speaker Detection</title>
<style>
* { box-sizing: border-box; }
body {
    margin: 0;
    background: #0b0b0b;
    color: white;
    font-family: Arial, sans-serif;
    text-align: center;
}
h1 { margin: 20px 0 15px 0; font-size: 28px; }
.container { width: 94%; max-width: 1280px; margin: auto; }
.video-wrapper {
    position: relative;
    width: 100%;
    background: black;
    border-radius: 10px;
    overflow: hidden;
    border: 2px solid #333;
}
#video { display: block; width: 100%; height: auto; }
#status {
    margin: 15px auto;
    padding: 14px 20px;
    width: 100%;
    background: #191919;
    border-radius: 8px;
    font-size: 18px;
}
#speaker { margin: 10px auto; font-size: 22px; font-weight: bold; }
#subtitle {
    margin: 15px auto;
    padding: 12px 20px;
    width: 100%;
    min-height: 45px;
    background: rgba(0,0,0,0.75);
    border-radius: 8px;
    font-size: 22px;
    font-weight: bold;
}
#state { margin-top: 8px; color: #aaa; font-size: 14px; }
</style>
</head>
<body>
<div class="container">
<h1>Smart Meeting — Active Speaker Detection</h1>
<div class="video-wrapper">
<img id="video" src="/video_feed" />
</div>
<div id="status">Loading...</div>
<div id="speaker">Active Speaker: None</div>
<div id="subtitle">No speech detected</div>
<div id="state">Camera and AI system starting...</div>
</div>
<script>
async function updateStatus() {
    try {
        const response = await fetch("/status");
        const data = await response.json();

        document.getElementById("status").innerText =
            "Persons Detected: " + data.persons_detected;

        document.getElementById("speaker").innerText =
            "Active Speaker: " +
            (data.active_speaker_id !== null
                ? "Person " + data.active_speaker_id
                : "None");

        document.getElementById("state").innerText =
            data.is_speaking ? "Speaking detected" : "Listening...";
    } catch (error) {
        document.getElementById("state").innerText = "Connection error";
    }
}

async function updateSubtitle() {
    try {
        const response = await fetch("/subtitle_text");
        const data = await response.json();

        document.getElementById("subtitle").innerText =
            data.text || "No speech detected";
    } catch (error) {
        // Ignore temporary errors.
    }
}

setInterval(updateStatus, 500);
setInterval(updateSubtitle, 500);

updateStatus();
updateSubtitle();
</script>
</body>
</html>
"""


# ============================================================
# VIDEO FEED
# ============================================================

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# STATUS API
# ============================================================

@app.route("/status")
def status():
    with state_lock:
        return jsonify(latest_status)


# ============================================================
# SUBTITLE API
# ============================================================

@app.route("/subtitle_text")
def subtitle_text():
    with state_lock:
        return jsonify({"text": latest_subtitle})


# ============================================================
# TRANSCRIPT API
# ============================================================

@app.route("/transcript")
def transcript():
    try:
        return jsonify(director.subtitle_engine.get_history())
    except Exception:
        return jsonify([])


# ============================================================
# MAIN
# ============================================================

def main():
    global director

    print()
    print("========================================")
    print(" SMART MEETING ACTIVE SPEAKER SYSTEM")
    print("========================================")
    print()
    print("Initializing SmartMeetingDirector...")

    director = SmartMeetingDirector()

    print("[Audio] Starting microphone...")

    audio_stream = sd.InputStream(
        samplerate=AUDIO_SAMPLE_RATE,
        channels=AUDIO_CHANNELS,
        dtype="float32",
        callback=audio_callback,
        blocksize=AUDIO_BLOCKSIZE
    )

    audio_stream.start()

    print("[Audio] Microphone started.")
    print("[Camera] Opening webcam...")

    camera_thread = threading.Thread(
        target=camera_loop, daemon=True, name="CameraCapture"
    )
    camera_thread.start()

    director_thread = threading.Thread(
        target=director_loop, daemon=True, name="DirectorAI"
    )
    director_thread.start()

    render_thread = threading.Thread(
        target=render_loop, daemon=True, name="VideoRenderer"
    )
    render_thread.start()

    analysis_thread = threading.Thread(
        target=analysis_loop, daemon=True, name="Analysis"
    )
    analysis_thread.start()

    print("[Processing] Camera + tracking + rendering started.")
    print("[Analysis] TalkNet + Whisper started.")
    print()
    print("========================================")
    print(" WEB SERVER")
    print("========================================")
    print(f"http://{SERVER_HOST}:{SERVER_PORT}")
    print()
    print("Open the URL above in your browser.")
    print()

    try:
        app.run(
            host=SERVER_HOST,
            port=SERVER_PORT,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    finally:
        try:
            audio_stream.stop()
            audio_stream.close()
        except Exception:
            pass


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
