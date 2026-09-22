import threading
import time
from collections import deque

import cv2
import numpy as np
import sounddevice as sd

from flask import Flask, Response, jsonify, request

from smart_meeting.config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    TARGET_FPS,
    TARGET_WIDTH,
    TARGET_HEIGHT,
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    AUDIO_BLOCKSIZE,
    WINDOW_FRAMES,
    WINDOW_AUDIO_SAMPLES,
    WHISPER_INTERVAL,
    WHISPER_MIN_AUDIO_SECONDS,
    WHISPER_AUDIO_PEAK_THRESHOLD,
    SERVER_HOST,
    SERVER_PORT,
)

import smart_meeting.config as _config
from smart_meeting.speaker_engine import SmartMeetingDirector


app = Flask(__name__)

# Set to True to print audio peak/rms values 4 times a second in the terminal.
DEBUG_ANALYSIS = False

MAX_NAME_LENGTH = 40

# Extra seconds trimmed from the END of the audio given to TalkNet, to
# compensate for webcam lag. Optional key in config.py.
AV_AUDIO_TRIM_SEC = getattr(_config, "AV_AUDIO_TRIM_SEC", 0.0)

# With exactly one person in view and clear voice activity, credit that
# person even if TalkNet is unsure (avoids "Unknown Speaker").
SINGLE_PERSON_FALLBACK = getattr(_config, "SINGLE_PERSON_FALLBACK", True)
VOICE_HOLD_SECONDS = 1.0

# Longest audio chunk sent to Whisper in one go.
WHISPER_MAX_SEGMENT_SECONDS = 8.0
WHISPER_OVERLAP_SECONDS = 0.3

# The speaker decision trails the real voice by roughly this long (1 second
# analysis window + hysteresis). Used to place speaker changes correctly.
SPEAKER_DETECT_LAG_SEC = 1.0
MIN_SPEAKER_RUN_SECONDS = 1.0

frame_lock = threading.Lock()
audio_lock = threading.Lock()
state_lock = threading.Lock()
recording_lock = threading.Lock()

latest_frame = None
latest_frame_ts = 0.0
latest_frame_seq = 0
latest_output_frame = None
latest_subtitle = ""

latest_status = {
    "persons_detected": 0,
    "active_speaker_id": None,
    "speaking_score": 0.0,
    "is_speaking": False,
    "attendees": []
}

audio_buffer = []
audio_total_samples = 0
MAX_AUDIO_BUFFER = WINDOW_AUDIO_SAMPLES * 4

video_buffer = []
MAX_VIDEO_BUFFER = WINDOW_FRAMES * 4

director = None

# ============================================================
# MEETING ANALYTICS STATE
# ============================================================

meeting_start_time = None
speaker_seconds = {}
speaker_names = {}
speaker_change_count = 0
last_active_id_for_change_tracking = None

# (timestamp, active_id) samples so Whisper can name who spoke.
active_history = deque(maxlen=600)

# ============================================================
# RECORDING STATE
# ============================================================

recording_active = False
video_writer = None
recording_path = None


def audio_callback(indata, frames, time_info, status):
    global audio_total_samples
    if status:
        print("[Audio]", status)
    if indata is None:
        return
    try:
        samples = indata[:, 0].astype(np.float32).copy()
        with audio_lock:
            audio_buffer.extend(samples.tolist())
            audio_total_samples += len(samples)
            if len(audio_buffer) > MAX_AUDIO_BUFFER:
                del audio_buffer[: len(audio_buffer) - MAX_AUDIO_BUFFER]
    except Exception as exc:
        print("[Audio] Callback error:", repr(exc))


def camera_loop():
    global latest_frame, latest_frame_ts, latest_frame_seq
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
                latest_frame_ts = timestamp
                latest_frame_seq += 1
                video_buffer.append((timestamp, frame.copy()))
                if len(video_buffer) > MAX_VIDEO_BUFFER:
                    del video_buffer[: len(video_buffer) - MAX_VIDEO_BUFFER]
            time.sleep(0.001)
    finally:
        cap.release()
        print("[Camera] Webcam released.")


def director_loop():
    print("[Director] AI processing thread started.")

    last_seq = -1

    while True:
        # Only process each camera frame once. Re-processing the same frame
        # would fill the face history with duplicate images and hide the lip
        # movement TalkNet needs to see.
        with frame_lock:
            if latest_frame is None or latest_frame_seq == last_seq:
                frame = None
            else:
                frame = latest_frame.copy()
                frame_ts = latest_frame_ts
                last_seq = latest_frame_seq

        if frame is None:
            time.sleep(0.005)
            continue

        try:
            director.process_frame(frame, frame_ts)
        except Exception as exc:
            print("[Director] Frame error:", repr(exc))


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

            # ------------------------------------------------
            # Recording: write the rendered frame if active.
            # ------------------------------------------------

            with recording_lock:
                if recording_active and video_writer is not None:
                    try:
                        write_frame = rendered

                        if (
                            write_frame.shape[1] != TARGET_WIDTH
                            or write_frame.shape[0] != TARGET_HEIGHT
                        ):
                            write_frame = cv2.resize(
                                write_frame,
                                (TARGET_WIDTH, TARGET_HEIGHT),
                            )

                        video_writer.write(write_frame)
                    except Exception as exc:
                        print("[Recording] Write error:", repr(exc))

        except Exception as exc:
            print("[Render] Frame error:", repr(exc))


def analysis_loop():
    global latest_status
    global speaker_change_count
    global last_active_id_for_change_tracking

    print("[Analysis] Waiting for enough audio/video...")

    last_loop_time = time.time()
    last_real_speaker = None
    last_voice_time = 0.0

    while True:
        time.sleep(0.25)

        now_loop = time.time()
        # Cap the credit so one slow iteration cannot add many seconds of
        # talk time to whoever happens to be active afterwards.
        elapsed = min(now_loop - last_loop_time, 1.0)
        last_loop_time = now_loop

        with audio_lock:
            if len(audio_buffer) < WINDOW_AUDIO_SAMPLES:
                continue
            audio = np.asarray(audio_buffer[-WINDOW_AUDIO_SAMPLES:], dtype=np.float32)

        audio_snapshot_time = time.time()

        with frame_lock:
            if latest_frame is None:
                continue
            if video_buffer:
                window_end_time = video_buffer[-1][0]
            else:
                window_end_time = audio_snapshot_time

        # The newest audio is "now", but the newest video frame was captured
        # a moment ago. Drop that difference from the end of the audio so
        # both cover the same instant.
        lag = max(0.0, audio_snapshot_time - window_end_time) + AV_AUDIO_TRIM_SEC
        trim = int(min(lag, 0.5) * AUDIO_SAMPLE_RATE)
        if trim > 0 and trim < len(audio) // 2:
            talk_audio = audio[: len(audio) - trim]
        else:
            talk_audio = audio

        active_id = None
        score = 0.0

        try:
            active_id, score = director.evaluate_active_speaker(talk_audio, AUDIO_SAMPLE_RATE, window_end_time)
        except Exception as exc:
            print("[TalkNet] Analysis error:", repr(exc))

        try:
            status = director.get_status_summary()
        except Exception as exc:
            print("[Status] Error:", repr(exc))
            status = None

        # --------------------------------------------------
        # One person in view + clear voice = that person.
        # --------------------------------------------------

        recent = audio[-int(0.5 * AUDIO_SAMPLE_RATE):]
        if float(np.max(np.abs(recent))) > WHISPER_AUDIO_PEAK_THRESHOLD:
            last_voice_time = now_loop

        voice_recent = (now_loop - last_voice_time) < VOICE_HOLD_SECONDS
        fallback_used = False

        if SINGLE_PERSON_FALLBACK and active_id is None and voice_recent and status is not None:
            visible = [
                a["id"]
                for a in status.get("attendees", [])
                if isinstance(a, dict) and "id" in a and a.get("missed_frames", 0) <= 3
            ]
            if len(visible) == 1:
                active_id = visible[0]
                fallback_used = True

        # --------------------------------------------------
        # Track speaker changes + accumulate talk time.
        # --------------------------------------------------

        with state_lock:
            if active_id is not None:
                # A change is a different person taking over, not the same
                # person starting again after a pause.
                if last_real_speaker is not None and active_id != last_real_speaker:
                    speaker_change_count += 1
                last_real_speaker = active_id

                speaker_seconds[active_id] = speaker_seconds.get(active_id, 0.0) + elapsed

            last_active_id_for_change_tracking = active_id
            active_history.append((now_loop, active_id))

        if DEBUG_ANALYSIS:
            peak = float(np.max(np.abs(audio)))
            rms = float(np.sqrt(np.mean(audio ** 2)))
            print(
                "[DEBUG] peak={:.4f} rms={:.4f} talknet_score={:.2f} active={} fallback={}".format(
                    peak, rms, float(score), active_id, fallback_used
                )
            )

        if status is not None:
            status["speaking_score"] = round(float(score), 3)

            if fallback_used:
                status["active_speaker_id"] = active_id
                status["is_speaking"] = True
                director.set_fallback_speaker(active_id)

            with state_lock:
                latest_status = status


def speaker_runs(start_time, end_time):
    """
    Split a time range into consecutive runs of one speaker each, so a chunk
    of audio that contains two people is transcribed and labelled separately.
    Returns a list of (name, run_start, run_end).
    """
    with state_lock:
        samples = [
            (ts - SPEAKER_DETECT_LAG_SEC, sid)
            for ts, sid in active_history
            if start_time <= ts - SPEAKER_DETECT_LAG_SEC <= end_time
        ]
        names = dict(speaker_names)

    runs = []  # [speaker_id, run_start, run_end]

    for ts, sid in samples:
        if sid is None:
            if runs:
                runs[-1][2] = ts
            continue

        if runs and runs[-1][0] == sid:
            runs[-1][2] = ts
        elif runs:
            runs[-1][2] = ts
            runs.append([sid, ts, ts])
        else:
            runs.append([sid, start_time, ts])

    if not runs:
        return [("Unknown Speaker", start_time, end_time)]

    runs[0][1] = start_time
    runs[-1][2] = end_time

    # Absorb very short runs into their neighbour.
    merged = []
    for run in runs:
        if merged and merged[-1][0] == run[0]:
            merged[-1][2] = run[2]
        elif merged and (run[2] - run[1]) < MIN_SPEAKER_RUN_SECONDS:
            merged[-1][2] = run[2]
        else:
            merged.append(run)

    if len(merged) > 1 and (merged[0][2] - merged[0][1]) < MIN_SPEAKER_RUN_SECONDS:
        merged[1][1] = merged[0][1]
        merged.pop(0)

    return [
        (names.get(sid, "Person {}".format(sid)), t0, t1)
        for sid, t0, t1 in merged
    ]


def whisper_loop():
    """
    Runs Whisper on its own thread so slow transcription never freezes
    speaker detection, and transcribes every stretch of audio once (plus a
    small overlap) instead of only the last few seconds. A chunk that holds
    more than one speaker is split so each line gets the right name.
    """
    global latest_subtitle

    print("[Whisper] Subtitle thread started.")

    sr = AUDIO_SAMPLE_RATE
    min_new = int(WHISPER_MIN_AUDIO_SECONDS * sr)
    max_segment = int(WHISPER_MAX_SEGMENT_SECONDS * sr)
    overlap = int(WHISPER_OVERLAP_SECONDS * sr)
    min_piece = int(1.0 * sr)

    last_consumed = 0
    last_run = 0.0

    while True:
        time.sleep(0.25)

        if (time.time() - last_run) < WHISPER_INTERVAL:
            continue

        with audio_lock:
            total = audio_total_samples
            new_samples = total - last_consumed
            if new_samples < min_new or not audio_buffer:
                continue
            take = min(new_samples + overlap, max_segment, len(audio_buffer))
            segment = np.asarray(audio_buffer[-take:], dtype=np.float32)

        segment_end = time.time()
        segment_start = segment_end - take / float(sr)
        last_consumed = total

        if float(np.max(np.abs(segment))) <= WHISPER_AUDIO_PEAK_THRESHOLD:
            last_run = time.time()
            continue

        for name, t0, t1 in speaker_runs(segment_start, segment_end):
            i0 = int(max(0.0, t0 - segment_start) * sr)
            i1 = int(min(take, (t1 - segment_start) * sr))
            piece = segment[i0:i1]

            if len(piece) < min_piece:
                continue
            if float(np.max(np.abs(piece))) <= WHISPER_AUDIO_PEAK_THRESHOLD:
                continue

            try:
                subtitle = director.subtitle_engine.transcribe_audio_segment(piece, sr, name)
            except Exception as exc:
                print("[Whisper] Error:", repr(exc))
                subtitle = ""

            if subtitle:
                with state_lock:
                    latest_subtitle = subtitle

        last_run = time.time()


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

        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n")

        time.sleep(1.0 / max(1, TARGET_FPS))


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    response = Response(DASHBOARD_HTML, mimetype="text/html")
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with state_lock:
        return jsonify(latest_status)


@app.route("/subtitle_text")
def subtitle_text():
    with state_lock:
        return jsonify({"text": latest_subtitle})


@app.route("/transcript")
def transcript():
    try:
        return jsonify(director.subtitle_engine.get_history())
    except Exception:
        return jsonify([])


@app.route("/meeting_stats")
def meeting_stats():
    with state_lock:
        if meeting_start_time is not None:
            duration = time.time() - meeting_start_time
        else:
            duration = 0.0

        speakers = []
        for sid, secs in speaker_seconds.items():
            speakers.append({
                "id": sid,
                "name": speaker_names.get(sid, "Person {}".format(sid)),
                "seconds": round(secs, 1),
            })

        speakers.sort(key=lambda item: -item["seconds"])

        # Every custom name, including people who have not spoken yet.
        names = {str(sid): name for sid, name in speaker_names.items()}

        changes = speaker_change_count
        participants = latest_status.get("persons_detected", 0)

    mins = int(duration) // 60
    secs_part = int(duration) % 60
    duration_formatted = "{:02d}:{:02d}".format(mins, secs_part)

    with recording_lock:
        is_recording = recording_active

    return jsonify({
        "duration_seconds": int(duration),
        "duration_formatted": duration_formatted,
        "speaker_changes": changes,
        "participants": participants,
        "speaker_times": speakers,
        "names": names,
        "recording": is_recording,
    })


@app.route("/rename_speaker", methods=["POST"])
def rename_speaker():
    try:
        data = request.get_json(force=True)
        raw_id = data.get("id")
        name = str(data.get("name", "")).strip()[:MAX_NAME_LENGTH]

        if raw_id is None or not name:
            return jsonify({"ok": False, "error": "Missing id or name."}), 400

        speaker_id = int(raw_id)

        with state_lock:
            speaker_names[speaker_id] = name

        return jsonify({"ok": True, "name": name})

    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/download_transcript")
def download_transcript():
    try:
        history = director.subtitle_engine.get_history()
    except Exception:
        history = []

    lines = []
    lines.append("Smart Meeting Transcript")
    lines.append("=" * 40)
    lines.append("")

    if not history:
        lines.append("No speech was recorded during this meeting.")
    else:
        for entry in history:
            speaker = entry.get("speaker", "Unknown")
            text_value = entry.get("text", "")
            time_value = entry.get("time", "")
            lines.append("[{}] {}: {}".format(time_value, speaker, text_value))

    content = "\n".join(lines)

    return Response(
        content,
        mimetype="text/plain",
        headers={"Content-Disposition": "attachment; filename=meeting_transcript.txt"},
    )


@app.route("/recording/start", methods=["POST"])
def recording_start():
    global recording_active, video_writer, recording_path

    with recording_lock:
        if recording_active:
            return jsonify({"ok": True, "already_recording": True, "path": recording_path})

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        recording_path = "meeting_recording_{}.avi".format(timestamp_str)

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_writer = cv2.VideoWriter(
            recording_path,
            fourcc,
            TARGET_FPS,
            (TARGET_WIDTH, TARGET_HEIGHT),
        )

        if not video_writer.isOpened():
            video_writer = None
            recording_path = None
            return jsonify({"ok": False, "error": "Could not open video writer."}), 500

        recording_active = True

    print("[Recording] Started:", recording_path)

    return jsonify({"ok": True, "path": recording_path})


@app.route("/recording/stop", methods=["POST"])
def recording_stop():
    global recording_active, video_writer

    with recording_lock:
        recording_active = False

        if video_writer is not None:
            try:
                video_writer.release()
            except Exception as exc:
                print("[Recording] Release error:", repr(exc))
            video_writer = None

    print("[Recording] Stopped.")

    return jsonify({"ok": True})


# ============================================================
# MAIN
# ============================================================

def main():
    global director
    global meeting_start_time

    print()
    print("========================================")
    print(" SMART MEETING ACTIVE SPEAKER SYSTEM")
    print("========================================")
    print()
    print("Initializing SmartMeetingDirector...")

    director = SmartMeetingDirector()

    meeting_start_time = time.time()

    print("[Audio] Starting microphone...")

    audio_device = getattr(_config, "AUDIO_DEVICE", None)

    try:
        if audio_device is None:
            default_in = sd.default.device[0]
            device_info = sd.query_devices(default_in)
            print(
                f"[Audio] Using DEFAULT input device #{default_in}: "
                f"{device_info['name']}"
            )
        else:
            device_info = sd.query_devices(audio_device)
            print(
                f"[Audio] Using CONFIGURED input device #{audio_device}: "
                f"{device_info['name']}"
            )
    except Exception as exc:
        print("[Audio] Could not read device info:", repr(exc))
        print("[Audio] Full device list:")
        print(sd.query_devices())

    try:
        audio_stream = sd.InputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            channels=AUDIO_CHANNELS,
            dtype="float32",
            callback=audio_callback,
            blocksize=AUDIO_BLOCKSIZE,
            device=audio_device,
        )
        audio_stream.start()
    except Exception as exc:
        print("[Audio] FAILED to start microphone:", repr(exc))
        print("[Audio] Available devices:")
        print(sd.query_devices())
        print(
            "[Audio] Set AUDIO_DEVICE in smart_meeting/config.py to one "
            "of the input device numbers above, then restart."
        )
        raise

    print("[Audio] Microphone started.")
    print("[Camera] Opening webcam...")

    camera_thread = threading.Thread(target=camera_loop, daemon=True, name="CameraCapture")
    camera_thread.start()

    director_thread = threading.Thread(target=director_loop, daemon=True, name="DirectorAI")
    director_thread.start()

    render_thread = threading.Thread(target=render_loop, daemon=True, name="VideoRenderer")
    render_thread.start()

    analysis_thread = threading.Thread(target=analysis_loop, daemon=True, name="Analysis")
    analysis_thread.start()

    whisper_thread = threading.Thread(target=whisper_loop, daemon=True, name="Whisper")
    whisper_thread.start()

    print("[Processing] Camera + tracking + rendering started.")
    print("[Analysis] TalkNet + Whisper started.")
    print()
    print("========================================")
    print(" WEB SERVER")
    print("========================================")
    print("http://{}:{}".format(SERVER_HOST, SERVER_PORT))
    print()
    print("Open the URL above in your browser.")
    print()

    try:
        app.run(
            host=SERVER_HOST,
            port=SERVER_PORT,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    finally:
        try:
            audio_stream.stop()
            audio_stream.close()
        except Exception:
            pass

        with recording_lock:
            if video_writer is not None:
                try:
                    video_writer.release()
                except Exception:
                    pass


# ============================================================
# FRONTEND (single-page dashboard, served at "/")
# Edit the HTML/CSS/JS below to change the UI. Restart app.py after editing.
# ============================================================

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart Meeting</title>
<style>
:root {
  --bg: #0e1219;
  --panel: #151b26;
  --raised: #1c2433;
  --raised-hover: #243044;
  --line: #252e40;
  --text: #e8ebf2;
  --muted: #8992a7;
  --signal: #ffb547;
  --signal-soft: rgba(255, 181, 71, 0.13);
  --ok: #5ec4b6;
  --rec: #ef5b5b;
  color-scheme: dark;
}

* { box-sizing: border-box; }
[hidden] { display: none !important; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 15px/1.45 "Segoe UI Variable Text", "Segoe UI", system-ui, -apple-system, Roboto, Helvetica, Arial, sans-serif;
  font-variant-numeric: tabular-nums;
}

button, input { font: inherit; color: inherit; }
:focus-visible { outline: 2px solid var(--signal); outline-offset: 2px; }

.shell { max-width: 1440px; margin: 0 auto; padding: 20px 24px 40px; }

/* ---------- Top bar ---------- */
.topbar {
  display: flex; align-items: center; justify-content: space-between;
  gap: 16px; flex-wrap: wrap; margin-bottom: 20px;
}
.brand h1 { margin: 0; font-size: 22px; font-weight: 650; letter-spacing: -0.01em; }
.brand p { margin: 2px 0 0; color: var(--muted); font-size: 13.5px; }
.top-right { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }

.clock { color: var(--muted); font-size: 14px; margin-right: 6px; }
.clock strong { color: var(--text); font-weight: 600; font-size: 16px; margin-left: 4px; }

.pill {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 6px 13px; border-radius: 999px;
  background: var(--panel); border: 1px solid var(--line);
  font-size: 13px; font-weight: 600;
}
.pill .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
.pill[data-state="live"] .dot { background: var(--ok); animation: pulse 1.8s ease-out infinite; }
.pill[data-state="offline"] .dot { background: var(--rec); }
.pill.rec .dot { background: var(--rec); animation: pulse-rec 1.4s ease-out infinite; }

@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(94, 196, 182, 0.55); }
  100% { box-shadow: 0 0 0 8px rgba(94, 196, 182, 0); }
}
@keyframes pulse-rec {
  0% { box-shadow: 0 0 0 0 rgba(239, 91, 91, 0.55); }
  100% { box-shadow: 0 0 0 8px rgba(239, 91, 91, 0); }
}

/* ---------- Layout ---------- */
.layout {
  display: grid; grid-template-columns: minmax(0, 1fr) 380px;
  gap: 20px; align-items: start;
}
.stage, .side { display: flex; flex-direction: column; gap: 20px; min-width: 0; }

.panel {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 14px; overflow: hidden;
}
.panel-head {
  display: flex; align-items: center; justify-content: space-between;
  gap: 12px; padding: 14px 18px 4px;
}
.panel-head h2 { margin: 0; font-size: 15px; font-weight: 650; }
.panel-head .aside { color: var(--muted); font-size: 13px; }
.panel-body { padding: 10px 18px 16px; }
.empty { margin: 0; padding: 10px 0 4px; color: var(--muted); font-size: 14px; max-width: 46ch; }

/* ---------- Video ---------- */
.video-card {
  border-radius: 18px; overflow: hidden; background: var(--panel);
  border: 1px solid var(--line);
  transition: border-color .2s, box-shadow .2s;
}
.video-card.has-speaker { border-color: var(--signal); box-shadow: 0 0 0 3px var(--signal-soft); }

.video-frame {
  position: relative; width: 100%; aspect-ratio: 16 / 9; max-height: 72vh;
  background: #05070a;
}
.video-frame img { display: block; width: 100%; height: 100%; object-fit: contain; }

.speaker-chip {
  position: absolute; top: 14px; left: 14px;
  display: inline-flex; align-items: center; gap: 10px;
  padding: 7px 14px 7px 12px; border-radius: 999px;
  background: rgba(14, 18, 25, 0.8);
  -webkit-backdrop-filter: blur(6px); backdrop-filter: blur(6px);
  border: 1px solid rgba(255, 181, 71, 0.5);
  font-weight: 600; font-size: 14px;
}
.speaker-chip .score { color: var(--muted); font-weight: 500; font-size: 13px; }

.caption {
  padding: 14px 20px; min-height: 60px; display: flex; align-items: center;
  border-top: 1px solid var(--line);
  font-size: 20px; font-weight: 600; line-height: 1.35;
}
.caption.idle { color: var(--muted); font-weight: 400; font-size: 16px; }

/* ---------- Equalizer (speaking indicator) ---------- */
.eq { display: inline-flex; align-items: flex-end; gap: 2px; height: 14px; width: 14px; flex: none; }
.eq span {
  flex: 1; height: 100%; border-radius: 1px; background: var(--muted);
  transform: scaleY(0.3); transform-origin: bottom;
}
.eq.on span { background: var(--signal); animation: eq 0.9s ease-in-out infinite; }
.eq.on span:nth-child(2) { animation-delay: -0.3s; }
.eq.on span:nth-child(3) { animation-delay: -0.6s; }
@keyframes eq { 0%, 100% { transform: scaleY(0.3); } 50% { transform: scaleY(1); } }

/* ---------- Participants ---------- */
.people { list-style: none; margin: 0; padding: 0; }
.person {
  display: grid; grid-template-columns: 36px minmax(0, 1fr) auto;
  align-items: center; gap: 12px;
  padding: 9px 10px; margin: 0 -10px; border-radius: 10px;
  transition: background .15s;
}
.person.active { background: var(--signal-soft); box-shadow: inset 3px 0 0 var(--signal); }
.person.away { opacity: 0.55; }
.avatar {
  width: 36px; height: 36px; border-radius: 50%;
  display: grid; place-items: center;
  color: #10141c; font-weight: 700; font-size: 14px;
}
.who { min-width: 0; }
.name-line { display: flex; align-items: center; gap: 6px; min-height: 24px; }
.name { font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.rename-input {
  width: 100%; min-width: 0; padding: 3px 8px;
  border-radius: 6px; border: 1px solid var(--signal); background: var(--bg);
}
.meta { color: var(--muted); font-size: 13px; }
.person.active .meta { color: var(--signal); }
.icon-btn {
  flex: none; width: 24px; height: 24px; display: grid; place-items: center;
  border: 0; border-radius: 6px; background: transparent; color: var(--muted);
  cursor: pointer; opacity: 0; transition: opacity .15s, background .15s;
}
.person:hover .icon-btn, .icon-btn:focus-visible { opacity: 1; }
.icon-btn:hover { background: var(--raised-hover); color: var(--text); }
@media (hover: none) { .icon-btn { opacity: 0.8; } }
.person-right { display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: 13px; }
.person-msg { margin: 8px 0 0; font-size: 13px; color: var(--rec); min-height: 0; }

/* ---------- Stats strip ---------- */
.strip { display: grid; grid-template-columns: repeat(3, 1fr); }
.strip > div { padding: 14px 18px; }
.strip > div + div { border-left: 1px solid var(--line); }
.strip .v { font-size: 24px; font-weight: 650; line-height: 1.1; }
.strip .l { color: var(--muted); font-size: 13px; margin-top: 3px; }

/* ---------- Speaking time ---------- */
.bar-row {
  display: grid; grid-template-columns: 96px minmax(0, 1fr) 92px;
  align-items: center; gap: 10px; padding: 6px 0; font-size: 14px;
}
.bar-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar-track { height: 10px; border-radius: 5px; background: var(--raised); overflow: hidden; }
.bar-fill { height: 100%; border-radius: 5px; transition: width .4s ease; }
.bar-value { text-align: right; color: var(--muted); font-size: 13px; }

/* ---------- Buttons / recording ---------- */
.btn {
  display: inline-flex; align-items: center; justify-content: center; gap: 9px;
  padding: 9px 16px; border-radius: 9px;
  border: 1px solid var(--line); background: var(--raised);
  font-weight: 600; font-size: 14px; cursor: pointer; text-decoration: none; color: var(--text);
  transition: background .15s, border-color .15s;
}
.btn:hover:not(:disabled) { background: var(--raised-hover); }
.btn:disabled { opacity: 0.55; cursor: default; }
.btn .glyph { width: 10px; height: 10px; background: var(--rec); border-radius: 50%; }
.btn.recording { background: var(--rec); border-color: var(--rec); color: #fff; }
.btn.recording:hover:not(:disabled) { background: #f57070; }
.btn.recording .glyph { background: #fff; border-radius: 2px; }
.rec-note { margin: 10px 0 0; color: var(--muted); font-size: 13px; max-width: 46ch; }

/* ---------- Transcript ---------- */
.transcript-list { max-height: 340px; overflow-y: auto; padding: 4px 18px 14px; }
.line {
  display: grid; grid-template-columns: 120px minmax(0, 1fr) auto;
  gap: 12px; padding: 9px 0; border-bottom: 1px solid var(--line);
  font-size: 14.5px; line-height: 1.45;
}
.line:last-child { border-bottom: 0; }
.line .spk { font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.line .tm { color: var(--muted); font-size: 12.5px; padding-top: 1px; }
.transcript-list .empty { padding-top: 8px; }

/* ---------- Responsive ---------- */
@media (max-width: 1020px) {
  .layout { grid-template-columns: minmax(0, 1fr); }
}
@media (max-width: 560px) {
  .shell { padding: 14px 14px 32px; }
  .line { grid-template-columns: minmax(0, 1fr) auto; }
  .line .spk { grid-column: 1 / -1; }
  .caption { font-size: 17px; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
</head>
<body>
<div class="shell">

  <header class="topbar">
    <div class="brand">
      <h1>Smart Meeting</h1>
      <p>Active speaker detection</p>
    </div>
    <div class="top-right">
      <div class="pill rec" id="rec-pill" hidden><span class="dot"></span>Recording</div>
      <div class="clock">Meeting time<strong id="duration">00:00</strong></div>
      <div class="pill" id="conn" data-state="connecting"><span class="dot"></span><span id="conn-text">Connecting</span></div>
    </div>
  </header>

  <div class="layout">

    <section class="stage">
      <div class="video-card" id="video-card">
        <div class="video-frame">
          <img id="video" src="/video_feed" alt="Live meeting video">
          <div class="speaker-chip" id="speaker-chip" hidden>
            <span class="eq" id="chip-eq"><span></span><span></span><span></span></span>
            <span id="chip-name"></span>
            <span class="score" id="chip-score"></span>
          </div>
        </div>
        <div class="caption idle" id="caption">Captions appear here when someone speaks.</div>
      </div>

      <div class="panel">
        <div class="panel-head">
          <h2>Transcript</h2>
          <a class="btn" href="/download_transcript" style="padding:6px 12px;font-size:13px">Download .txt</a>
        </div>
        <div class="transcript-list" id="transcript-list">
          <p class="empty">No speech yet. Lines are added here as people talk, newest first.</p>
        </div>
      </div>
    </section>

    <aside class="side">

      <div class="panel">
        <div class="panel-head">
          <h2>Participants</h2>
          <span class="aside" id="people-count">0 in view</span>
        </div>
        <div class="panel-body">
          <ul class="people" id="people"></ul>
          <p class="empty" id="people-empty">No one detected yet. Make sure the camera can see everyone's face.</p>
          <p class="person-msg" id="person-msg" hidden></p>
        </div>
      </div>

      <div class="panel strip">
        <div><div class="v" id="stat-inview">0</div><div class="l">In view</div></div>
        <div><div class="v" id="stat-speakers">0</div><div class="l">Have spoken</div></div>
        <div><div class="v" id="stat-changes">0</div><div class="l">Speaker changes</div></div>
      </div>

      <div class="panel">
        <div class="panel-head"><h2>Speaking time</h2><span class="aside" id="talk-total"></span></div>
        <div class="panel-body" id="talk-list">
          <p class="empty">Talk time shows up after the first person speaks.</p>
        </div>
      </div>

      <div class="panel">
        <div class="panel-head"><h2>Recording</h2></div>
        <div class="panel-body">
          <button class="btn" id="rec-btn" type="button"><span class="glyph"></span><span id="rec-btn-text">Start recording</span></button>
          <p class="rec-note" id="rec-note">Saves the rendered video to the project folder. Audio is not included.</p>
        </div>
      </div>

    </aside>
  </div>
</div>

<script>
(function () {
  "use strict";

  const $ = function (id) { return document.getElementById(id); };

  const state = {
    status: null,
    statsNames: {},
    localNames: {},
    seconds: {},
    recording: false,
    recBusy: false,
    lastRecPath: null,
    editingId: null
  };

  /* ---------- helpers ---------- */

  function nameFor(id) {
    const k = String(id);
    return state.localNames[k] || state.statsNames[k] || "Person " + k;
  }

  function colorFor(id) {
    const hue = (Number(id) * 67 + 200) % 360;
    return "hsl(" + hue + ", 55%, 66%)";
  }

  function fmtSecs(s) {
    if (s < 60) return s.toFixed(1) + "s";
    const m = Math.floor(s / 60);
    const r = Math.floor(s % 60);
    return m + "m " + (r < 10 ? "0" + r : r) + "s";
  }

  function attendeeIds(list) {
    if (!Array.isArray(list)) return null;
    if (list.length === 0) return [];
    const ids = [];
    list.forEach(function (a) {
      let id = null;
      if (typeof a === "number") id = a;
      else if (typeof a === "string" && a.trim() !== "" && !isNaN(Number(a))) id = Number(a);
      else if (a && typeof a === "object") {
        id = a.id != null ? a.id : (a.track_id != null ? a.track_id : (a.person_id != null ? a.person_id : (a.speaker_id != null ? a.speaker_id : null)));
      }
      if (id !== null && id !== undefined && !isNaN(Number(id))) ids.push(Number(id));
    });
    return ids.length ? ids : null;
  }

  function getJSON(url) {
    return fetch(url, { cache: "no-store" }).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    });
  }

  /* Poll without overlapping requests. */
  function poll(fn, ms) {
    function tick() {
      Promise.resolve().then(fn).catch(function () {}).then(function () { setTimeout(tick, ms); });
    }
    tick();
  }

  /* ---------- connection pill ---------- */

  function setConn(online) {
    const pill = $("conn");
    pill.dataset.state = online ? "live" : "offline";
    $("conn-text").textContent = online ? "Live" : "Offline";
  }

  /* ---------- video ---------- */

  $("video").addEventListener("error", function () {
    setTimeout(function () { $("video").src = "/video_feed?t=" + Date.now(); }, 2000);
  });

  /* ---------- participants ---------- */

  const rows = new Map();

  function pencilSvg() {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("width", "14"); svg.setAttribute("height", "14");
    svg.setAttribute("viewBox", "0 0 24 24"); svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor"); svg.setAttribute("stroke-width", "2");
    svg.setAttribute("stroke-linecap", "round"); svg.setAttribute("stroke-linejoin", "round");
    const p = document.createElementNS(ns, "path");
    p.setAttribute("d", "M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z");
    svg.appendChild(p);
    return svg;
  }

  function createRow(id) {
    const li = document.createElement("li");
    li.className = "person";

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.style.background = colorFor(id);
    avatar.textContent = String(id);

    const who = document.createElement("div");
    who.className = "who";
    const nameLine = document.createElement("div");
    nameLine.className = "name-line";
    const nameEl = document.createElement("span");
    nameEl.className = "name";
    const editBtn = document.createElement("button");
    editBtn.type = "button";
    editBtn.className = "icon-btn";
    editBtn.appendChild(pencilSvg());
    nameLine.appendChild(nameEl);
    nameLine.appendChild(editBtn);
    const meta = document.createElement("div");
    meta.className = "meta";
    who.appendChild(nameLine);
    who.appendChild(meta);

    const right = document.createElement("div");
    right.className = "person-right";
    const eq = document.createElement("span");
    eq.className = "eq";
    eq.innerHTML = "<span></span><span></span><span></span>";
    const time = document.createElement("span");
    right.appendChild(eq);
    right.appendChild(time);

    li.appendChild(avatar);
    li.appendChild(who);
    li.appendChild(right);

    const row = { id: id, li: li, nameEl: nameEl, nameLine: nameLine, editBtn: editBtn, meta: meta, eq: eq, time: time, input: null };
    editBtn.addEventListener("click", function () { startEdit(row); });
    return row;
  }

  function startEdit(row) {
    if (state.editingId !== null) return;
    state.editingId = row.id;
    hideMsg();

    const input = document.createElement("input");
    input.type = "text";
    input.className = "rename-input";
    input.maxLength = 40;
    input.value = nameFor(row.id);
    input.setAttribute("aria-label", "Name for participant " + row.id);
    row.input = input;

    row.nameEl.hidden = true;
    row.editBtn.hidden = true;
    row.nameLine.insertBefore(input, row.nameEl);
    input.focus();
    input.select();

    let done = false;
    function finish(save) {
      if (done) return;
      done = true;
      const value = input.value.trim();
      input.remove();
      row.input = null;
      row.nameEl.hidden = false;
      row.editBtn.hidden = false;
      state.editingId = null;
      if (save && value && value !== nameFor(row.id)) {
        saveName(row.id, value);
      } else {
        renderParticipants();
      }
    }
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") { e.preventDefault(); finish(true); }
      else if (e.key === "Escape") { e.preventDefault(); finish(false); }
    });
    input.addEventListener("blur", function () { finish(false); });
  }

  function showMsg(text) {
    const el = $("person-msg");
    el.textContent = text;
    el.hidden = false;
    clearTimeout(showMsg.t);
    showMsg.t = setTimeout(hideMsg, 4000);
  }
  function hideMsg() { $("person-msg").hidden = true; }

  function saveName(id, name) {
    fetch("/rename_speaker", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: id, name: name })
    }).then(function (r) { return r.json().catch(function () { return {}; }); })
      .then(function (d) {
        if (d && d.ok) {
          state.localNames[String(id)] = name;
        } else {
          showMsg("Couldn't rename this person. Try again.");
        }
        renderParticipants();
        renderSpeakingTime();
        renderChip();
      })
      .catch(function () {
        showMsg("Lost connection to the server. Try again.");
        renderParticipants();
      });
  }

  function renderParticipants() {
    if (state.editingId !== null) return;

    const st = state.status || {};
    const att = attendeeIds(st.attendees);
    const info = {};
    if (Array.isArray(st.attendees)) {
      st.attendees.forEach(function (a) {
        if (a && typeof a === "object" && a.id != null) info[String(a.id)] = a;
      });
    }
    const activeId = (st.active_speaker_id !== null && st.active_speaker_id !== undefined) ? Number(st.active_speaker_id) : null;

    const ids = new Set();
    if (att) att.forEach(function (i) { ids.add(i); });
    Object.keys(state.seconds).forEach(function (k) { ids.add(Number(k)); });
    if (activeId !== null) ids.add(activeId);
    const sorted = Array.from(ids).sort(function (a, b) { return a - b; });

    const list = $("people");

    rows.forEach(function (row, id) {
      if (!ids.has(id)) { row.li.remove(); rows.delete(id); }
    });

    sorted.forEach(function (id, index) {
      let row = rows.get(id);
      if (!row) { row = createRow(id); rows.set(id, row); }

      const isActive = id === activeId;
      const inView = att ? att.indexOf(id) !== -1 : true;
      const secs = state.seconds[String(id)];

      row.li.classList.toggle("active", isActive);
      row.li.classList.toggle("away", !inView && !isActive);
      row.nameEl.textContent = nameFor(id);
      row.editBtn.setAttribute("aria-label", "Rename " + nameFor(id));

      let meta;
      if (isActive) {
        meta = st.speaking_score > 0 ? "Speaking, score " + Number(st.speaking_score).toFixed(2) : "Active speaker";
      } else if (!inView) {
        meta = "Out of view";
      } else {
        const ai = info[String(id)];
        meta = (ai && ai.score > -10) ? "Listening, score " + Number(ai.score).toFixed(2) : "Listening";
      }
      row.meta.textContent = meta;
      row.eq.classList.toggle("on", isActive && !!st.is_speaking);
      row.time.textContent = secs ? fmtSecs(secs) : "";

      if (list.children[index] !== row.li) list.insertBefore(row.li, list.children[index] || null);
    });

    $("people-empty").hidden = sorted.length > 0;
    const inViewCount = (st.persons_detected != null) ? st.persons_detected : sorted.length;
    $("people-count").textContent = inViewCount + " in view";
  }

  /* ---------- video chip + caption ---------- */

  function renderChip() {
    const st = state.status || {};
    const chip = $("speaker-chip");
    const card = $("video-card");
    const hasActive = st.active_speaker_id !== null && st.active_speaker_id !== undefined;

    chip.hidden = !hasActive;
    card.classList.toggle("has-speaker", hasActive);
    if (!hasActive) return;

    $("chip-name").textContent = nameFor(st.active_speaker_id);
    $("chip-score").textContent = st.speaking_score > 0 ? Number(st.speaking_score).toFixed(2) : "";
    $("chip-eq").classList.toggle("on", !!st.is_speaking);
  }

  function updateStatus() {
    return getJSON("/status").then(function (data) {
      state.status = data;
      setConn(true);
      $("stat-inview").textContent = data.persons_detected != null ? data.persons_detected : 0;
      renderChip();
      renderParticipants();
    }).catch(function (e) { setConn(false); throw e; });
  }

  function updateSubtitle() {
    return getJSON("/subtitle_text").then(function (data) {
      const cap = $("caption");
      if (data.text) {
        cap.textContent = data.text;
        cap.classList.remove("idle");
      } else {
        cap.textContent = "Captions appear here when someone speaks.";
        cap.classList.add("idle");
      }
    });
  }

  /* ---------- transcript ---------- */

  let lastTranscriptCount = 0;

  function updateTranscript() {
    return getJSON("/transcript").then(function (data) {
      if (!Array.isArray(data) || data.length === 0) return;
      if (data.length === lastTranscriptCount) return;
      lastTranscriptCount = data.length;

      const box = $("transcript-list");
      box.textContent = "";
      data.slice().reverse().forEach(function (entry) {
        const line = document.createElement("div");
        line.className = "line";
        const spk = document.createElement("span");
        spk.className = "spk";
        spk.textContent = entry.speaker || "Unknown";
        const txt = document.createElement("span");
        txt.textContent = entry.text || "";
        const tm = document.createElement("span");
        tm.className = "tm";
        tm.textContent = entry.time || "";
        line.appendChild(spk);
        line.appendChild(txt);
        line.appendChild(tm);
        box.appendChild(line);
      });
    });
  }

  /* ---------- meeting stats ---------- */

  function renderSpeakingTime() {
    const box = $("talk-list");
    const ids = Object.keys(state.seconds);
    if (ids.length === 0) {
      box.innerHTML = '<p class="empty">Talk time shows up after the first person speaks.</p>';
      $("talk-total").textContent = "";
      return;
    }

    let total = 0;
    ids.forEach(function (k) { total += state.seconds[k]; });
    ids.sort(function (a, b) { return state.seconds[b] - state.seconds[a]; });

    box.textContent = "";
    ids.forEach(function (k) {
      const secs = state.seconds[k];
      const pct = total > 0 ? (secs / total) * 100 : 0;

      const row = document.createElement("div");
      row.className = "bar-row";

      const label = document.createElement("div");
      label.className = "bar-label";
      label.textContent = nameFor(k);

      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      fill.style.width = pct.toFixed(1) + "%";
      fill.style.background = colorFor(k);
      track.appendChild(fill);

      const val = document.createElement("div");
      val.className = "bar-value";
      val.textContent = fmtSecs(secs) + "  " + Math.round(pct) + "%";

      row.appendChild(label);
      row.appendChild(track);
      row.appendChild(val);
      box.appendChild(row);
    });
    $("talk-total").textContent = "Total " + fmtSecs(total);
  }

  function updateMeetingStats() {
    return getJSON("/meeting_stats").then(function (data) {
      $("duration").textContent = data.duration_formatted || "00:00";
      $("stat-changes").textContent = data.speaker_changes != null ? data.speaker_changes : 0;

      state.statsNames = {};
      state.seconds = {};
      (data.speaker_times || []).forEach(function (s) {
        const k = String(s.id);
        state.statsNames[k] = s.name;
        state.seconds[k] = s.seconds;
        delete state.localNames[k];   /* server name wins once the person has spoken */
      });

      if (data.names && typeof data.names === "object") {
        Object.keys(data.names).forEach(function (k) {
          state.statsNames[k] = data.names[k];
          delete state.localNames[k];
        });
      }

      $("stat-speakers").textContent = (data.speaker_times || []).length;

      if (!state.recBusy) {
        if (state.recording && !data.recording) { /* stopped elsewhere */ }
        state.recording = !!data.recording;
      }

      renderSpeakingTime();
      renderParticipants();
      renderChip();
      renderRecording();
    });
  }

  /* ---------- recording ---------- */

  function renderRecording() {
    const btn = $("rec-btn");
    btn.classList.toggle("recording", state.recording);
    btn.disabled = state.recBusy;
    $("rec-btn-text").textContent = state.recording ? "Stop recording" : "Start recording";
    $("rec-pill").hidden = !state.recording;
  }

  function setRecNote(text) { $("rec-note").textContent = text; }

  $("rec-btn").addEventListener("click", function () {
    if (state.recBusy) return;
    const starting = !state.recording;
    state.recBusy = true;
    renderRecording();

    fetch(starting ? "/recording/start" : "/recording/stop", { method: "POST" })
      .then(function (r) {
        return r.json().catch(function () { return {}; }).then(function (d) { return { ok: r.ok, data: d }; });
      })
      .then(function (res) {
        if (!res.ok || res.data.ok === false) {
          setRecNote(res.data.error || "Recording failed. Check the terminal for details.");
          return;
        }
        state.recording = starting;
        if (starting) {
          state.lastRecPath = res.data.path || null;
          setRecNote(state.lastRecPath
            ? "Recording to " + state.lastRecPath + " in the project folder. Audio is not included."
            : "Recording in progress. Audio is not included.");
        } else {
          setRecNote(state.lastRecPath
            ? "Saved as " + state.lastRecPath + " in the project folder."
            : "Recording stopped.");
        }
      })
      .catch(function () { setRecNote("Lost connection to the server. Try again."); })
      .then(function () { state.recBusy = false; renderRecording(); });
  });

  /* ---------- start ---------- */

  renderRecording();
  poll(updateStatus, 500);
  poll(updateSubtitle, 500);
  poll(updateTranscript, 1000);
  poll(updateMeetingStats, 1000);
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()