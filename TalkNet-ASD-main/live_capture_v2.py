import sys
import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import subprocess
import pickle
import os
import shutil
import time
import traceback
import whisper

# ---------------- FFMPEG PATH FIX ----------------
# If ffmpeg.exe is bundled inside this project folder, add it to PATH so
# both the bare "ffmpeg" command and subprocess calls can find it.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_FFMPEG_DIR = os.path.join(_BASE_DIR, "ffmpeg_bin", "ffmpeg-9.0.1-essentials_build", "bin")
if os.path.isdir(_FFMPEG_DIR) and _FFMPEG_DIR not in os.environ["PATH"]:
    os.environ["PATH"] = _FFMPEG_DIR + os.pathsep + os.environ["PATH"]
    print(f"[DEBUG] Added ffmpeg dir to PATH: {_FFMPEG_DIR}")
else:
    print(f"[DEBUG] ffmpeg_bin folder not found at expected path: {_FFMPEG_DIR}")
    print("[DEBUG] Make sure ffmpeg.exe is installed and on PATH, or update _FFMPEG_DIR above.")

WINDOW_SECONDS = 3
FPS = 25
SAMPLE_RATE = 16000
VIDEO_FOLDER = "demo"
VIDEO_NAME = "live"
RAW_VIDEO_FILE = os.path.join(VIDEO_FOLDER, "live_raw.avi")
RAW_AUDIO_FILE = os.path.join(VIDEO_FOLDER, "live_raw.wav")
COMBINED_FILE = os.path.join(VIDEO_FOLDER, f"{VIDEO_NAME}.avi")
SAVE_PATH = os.path.join(VIDEO_FOLDER, VIDEO_NAME)

print("Loading Whisper (tiny model, for speed)...")
whisper_model = whisper.load_model("tiny")
print("Whisper loaded.")


def record_window(cap):
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(RAW_VIDEO_FILE, fourcc, FPS, (frame_width, frame_height))

    num_frames_needed = WINDOW_SECONDS * FPS
    frames_captured = 0
    last_frame = None
    audio_buffer = []

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[DEBUG] audio callback status: {status}")
        audio_buffer.append(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=audio_callback)

    print(f"Recording {WINDOW_SECONDS}s window...")
    stream.start()

    while frames_captured < num_frames_needed:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        last_frame = frame.copy()
        frames_captured += 1

        preview = frame.copy()
        cv2.putText(preview, "Recording...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Live Speaker Detection", preview)
        cv2.waitKey(1)

    stream.stop()
    stream.close()
    writer.release()

    if audio_buffer:
        audio_data = np.concatenate(audio_buffer, axis=0)
        # Quick sanity check: is the mic actually picking up sound, or is it silence?
        peak = int(np.abs(audio_data).max())
        print(f"[DEBUG] audio peak amplitude this window: {peak} (out of 32767)")
        if peak < 200:
            print("[DEBUG] WARNING: audio looks silent/very quiet. Check mic selection/volume.")
        sf.write(RAW_AUDIO_FILE, audio_data, SAMPLE_RATE)
    else:
        print("WARNING: no audio captured this window.")
        return None

    return last_frame


def combine_audio_video():
    if os.path.exists(COMBINED_FILE):
        os.remove(COMBINED_FILE)
    cmd = [
        "ffmpeg", "-y",
        "-i", RAW_VIDEO_FILE,
        "-i", RAW_AUDIO_FILE,
        "-c:v", "copy",
        "-c:a", "aac",
        "-strict", "experimental",
        COMBINED_FILE,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("---- ffmpeg combine ERROR ----")
        print(result.stderr[-1500:])
        return False
    return True


def run_talknet():
    cmd = [sys.executable, "demoTalkNet.py", "--videoName", VIDEO_NAME, "--videoFolder", VIDEO_FOLDER, "--skipVis"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("---- TalkNet ERROR ----")
        print(result.stderr[-2000:])
    return result.returncode == 0


def read_results():
    """Returns (best_track_idx, avg_scores, bbox, num_persons_detected)."""
    tracks_path = os.path.join(SAVE_PATH, "pywork", "tracks.pckl")
    scores_path = os.path.join(SAVE_PATH, "pywork", "scores.pckl")

    if not os.path.exists(tracks_path) or not os.path.exists(scores_path):
        print(f"[DEBUG] tracks/scores pckl not found at {tracks_path}")
        return None, None, None, 0

    with open(tracks_path, "rb") as f:
        tracks = pickle.load(f)
    with open(scores_path, "rb") as f:
        scores = pickle.load(f)

    num_persons = len(tracks) if tracks else 0
    print(f"[DEBUG] Number of faces/persons detected this window: {num_persons}")

    if not scores:
        return None, None, None, num_persons

    avg_scores = [float(np.mean(s)) for s in scores]
    for i, sc in enumerate(avg_scores):
        print(f"[DEBUG]   Person {i}: avg talking score = {sc:.3f}")

    best_track_idx = int(np.argmax(avg_scores))

    best_track = tracks[best_track_idx]['proc_track']
    avg_x = float(np.mean(best_track['x']))
    avg_y = float(np.mean(best_track['y']))
    avg_s = float(np.mean(best_track['s']))
    bbox = (avg_x, avg_y, avg_s)

    return best_track_idx, avg_scores, bbox, num_persons


def transcribe_audio():
    """Run Whisper on this window's raw audio to get the spoken text (forced to English)."""
    try:
        result = whisper_model.transcribe(RAW_AUDIO_FILE, fp16=False, language="en")
        text = result.get("text", "").strip()
        print(f"[DEBUG] Whisper raw result: '{text}'")
        return text
    except Exception as e:
        print(f"[DEBUG] Whisper error: {e}")
        traceback.print_exc()
        return ""


def draw_result_frame(frame, bbox, track_idx, text, num_persons):
    """Draw the active speaker's box + transcript text + person count on the frame."""
    display = frame.copy()
    h, w, _ = display.shape

    cv2.putText(display, f"Persons detected: {num_persons}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if bbox is not None:
        cx, cy, s = bbox
        x1 = int(max(cx - s, 0))
        y1 = int(max(cy - s, 0))
        x2 = int(min(cx + s, w))
        y2 = int(min(cy + s, h))
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(display, f"Speaking: Person {track_idx}", (x1, max(y1 - 10, 45)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(display, "No active speaker box", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if text:
        cv2.rectangle(display, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.putText(display, text[:80], (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Live Speaker Detection", display)
    cv2.waitKey(1)


def cleanup():
    if os.path.exists(SAVE_PATH):
        shutil.rmtree(SAVE_PATH)
    if os.path.exists(RAW_VIDEO_FILE):
        os.remove(RAW_VIDEO_FILE)


def main():
    os.makedirs(VIDEO_FOLDER, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam.")
        return

    print("Starting live speaker detection loop. Press Ctrl+C in this terminal to stop.")

    try:
        while True:
            last_frame = record_window(cap)
            if last_frame is None:
                continue

            ok = combine_audio_video()
            if not ok:
                print("[DEBUG] Skipping window: combine_audio_video failed.")
                cleanup()
                continue

            print("Processing window with TalkNet...")
            try:
                success = run_talknet()
            except Exception:
                print("[ERROR] run_talknet crashed:")
                traceback.print_exc()
                success = False

            bbox, track_idx, text, num_persons = None, None, "", 0
            if success:
                try:
                    best_track_idx, avg_scores, bbox, num_persons = read_results()
                    if best_track_idx is not None:
                        track_idx = best_track_idx
                        print(f">>> Active speaker: Person {track_idx} (scores: {avg_scores})")
                    else:
                        print(">>> No clear speaker detected this window.")
                except Exception:
                    print("[ERROR] read_results crashed:")
                    traceback.print_exc()
            else:
                print("TalkNet processing failed for this window.")

            print("Transcribing audio...")
            text = transcribe_audio()
            if text:
                print(f">>> Transcript: {text}")
            else:
                print(">>> No transcript this window.")

            draw_result_frame(last_frame, bbox, track_idx if track_idx is not None else -1, text, num_persons)

            if os.path.exists(RAW_AUDIO_FILE):
                os.remove(RAW_AUDIO_FILE)

            cleanup()

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
