import threading
import time

import cv2
import numpy as np
import sounddevice as sd

from flask import Flask, Response, jsonify

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

from smart_meeting.speaker_engine import SmartMeetingDirector


app = Flask(__name__)

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

audio_buffer = []
MAX_AUDIO_BUFFER = WINDOW_AUDIO_SAMPLES * 4

video_buffer = []
MAX_VIDEO_BUFFER = WINDOW_FRAMES * 4

director = None


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


def analysis_loop():
    global latest_subtitle
    global latest_status
    print("[Analysis] Waiting for enough audio/video...")
    last_whisper_time = 0.0
    while True:
        time.sleep(0.25)
        with audio_lock:
            if len(audio_buffer) < WINDOW_AUDIO_SAMPLES:
                continue
            audio = np.asarray(audio_buffer[-WINDOW_AUDIO_SAMPLES:], dtype=np.float32)
        with frame_lock:
            if latest_frame is None:
                continue
            current_frame = latest_frame.copy()
            if video_buffer:
                window_end_time = video_buffer[-1][0]
            else:
                window_end_time = time.time()
        active_id = None
        score = 0.0
        try:
            active_id, score = director.evaluate_active_speaker(audio, AUDIO_SAMPLE_RATE, window_end_time)
        except Exception as exc:
            print("[TalkNet] Analysis error:", repr(exc))
        subtitle = ""
        now = time.time()
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak_ok = peak > WHISPER_AUDIO_PEAK_THRESHOLD
        interval_ok = (now - last_whisper_time) >= WHISPER_INTERVAL
        will_call_whisper = peak_ok and interval_ok
        print("[DEBUG] peak={:.4f} rms={:.4f} peak_ok={} interval_ok={} will_call={}".format(peak, rms, peak_ok, interval_ok, will_call_whisper))
        if peak_ok and interval_ok:
            if active_id is not None:
                speaker_name = "Person {}".format(active_id)
            else:
                speaker_name = "Unknown Speaker"
            try:
                subtitle = director.subtitle_engine.transcribe_audio_segment(audio, AUDIO_SAMPLE_RATE, speaker_name)
                last_whisper_time = now
            except Exception as exc:
                print("[Whisper] Error:", repr(exc))
        try:
            status = director.get_status_summary()
            status["speaking_score"] = round(float(score), 3)
            with state_lock:
                latest_status = status
                if subtitle:
                    latest_subtitle = subtitle
        except Exception as exc:
            print("[Status] Error:", repr(exc))


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


@app.route("/")
def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Smart Meeting AI | Dashboard</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>&#127908;</text></svg>">
<style>
:root{--bg:#0a0e17;--panel:#0f1420;--panel-border:#1c2333;--text:#e7ebf3;--text-dim:#8b95a8;--accent-blue:#3b82f6;--accent-green:#22c55e;--accent-purple:#a855f7;--accent-orange:#f59e0b;--accent-red:#ef4444;--radius:12px;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,sans-serif;display:flex;min-height:100vh;font-size:14px;}
#sidebar{width:220px;background:var(--panel);border-right:1px solid var(--panel-border);display:flex;flex-direction:column;padding:20px 14px;flex-shrink:0;}
.brand{display:flex;align-items:center;gap:10px;padding:0 6px 22px 6px;border-bottom:1px solid var(--panel-border);margin-bottom:16px;}
.brand-icon{width:34px;height:34px;border-radius:8px;background:linear-gradient(135deg,var(--accent-blue),var(--accent-purple));display:flex;align-items:center;justify-content:center;font-weight:700;}
.brand-name{font-weight:600;font-size:15px;}
.brand-sub{font-size:11px;color:var(--text-dim);}
nav{display:flex;flex-direction:column;gap:2px;}
.nav-item{display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:8px;color:var(--text-dim);}
.nav-item.active{background:#152040;color:var(--accent-blue);}
.sidebar-footer{margin-top:auto;padding:14px 10px;border-top:1px solid var(--panel-border);color:var(--text-dim);font-size:12px;line-height:1.5;}
#main{flex:1;display:flex;flex-direction:column;min-width:0;}
header{display:flex;align-items:center;justify-content:space-between;padding:14px 24px;border-bottom:1px solid var(--panel-border);background:var(--panel);}
.header-right{display:flex;align-items:center;gap:16px;}
.status-pill{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--text-dim);}
.status-pill .dot{width:8px;height:8px;border-radius:50%;background:var(--accent-green);}
.status-pill .dot.bad{background:var(--accent-red);}
#content{padding:20px 24px;display:flex;flex-direction:column;gap:16px;overflow-y:auto;}
.row{display:grid;gap:16px;}
.row-top{grid-template-columns:2fr 1fr 1fr;}
.row-bottom{grid-template-columns:1.3fr 1fr;}
.card{background:var(--panel);border:1px solid var(--panel-border);border-radius:var(--radius);padding:16px;display:flex;flex-direction:column;}
.card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;}
.card-title{font-weight:600;font-size:14px;}
.badge-live{display:flex;align-items:center;gap:5px;background:#12241a;color:var(--accent-green);font-size:11px;font-weight:600;padding:4px 9px;border-radius:20px;}
.badge-live .dot{width:6px;height:6px;border-radius:50%;background:var(--accent-green);}
#video-wrap{position:relative;border-radius:8px;overflow:hidden;background:#000;aspect-ratio:16/9;}
#video-wrap img{width:100%;height:100%;object-fit:cover;display:block;}
.corner-tag{position:absolute;top:10px;left:10px;background:var(--accent-red);color:#fff;font-size:11px;font-weight:700;padding:3px 8px;border-radius:4px;}
#speaker-photo{width:100%;aspect-ratio:4/5;background:#000;border-radius:8px;overflow:hidden;margin-bottom:10px;}
#speaker-photo img{width:100%;height:100%;object-fit:cover;}
#speaker-name{font-weight:700;font-size:16px;margin-bottom:4px;}
#speaker-status{display:flex;align-items:center;gap:6px;color:var(--accent-green);font-size:12px;margin-bottom:10px;}
.stat-row{display:flex;justify-content:space-between;font-size:12px;color:var(--text-dim);}
.stat-row b{color:var(--text);font-size:14px;display:block;}
.participant{display:flex;align-items:center;gap:10px;padding:8px 4px;border-radius:8px;}
.avatar{width:38px;height:38px;border-radius:50%;background:#1c2333;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:600;}
.p-name{font-weight:600;font-size:13px;}
.p-status{font-size:11px;color:var(--text-dim);display:flex;align-items:center;gap:5px;}
.p-status .dot{width:6px;height:6px;border-radius:50%;background:var(--text-dim);}
.p-status.speaking{color:var(--accent-green);}
.p-status.speaking .dot{background:var(--accent-green);}
.p-pct{margin-left:auto;font-size:12px;color:var(--text-dim);}
#transcript{flex:1;overflow-y:auto;max-height:360px;display:flex;flex-direction:column;gap:12px;padding-right:4px;}
.t-line{display:flex;gap:10px;font-size:13px;}
.t-time{color:var(--text-dim);font-size:11px;min-width:46px;padding-top:2px;}
.t-body b{display:block;font-size:12px;margin-bottom:2px;}
.t-body span{color:#c7cede;line-height:1.4;}
.t-empty{color:var(--text-dim);font-size:13px;padding:8px 0;}
.stat-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:16px;}
.stat-box{background:var(--bg);border:1px solid var(--panel-border);border-radius:8px;padding:12px;}
.stat-box .label{font-size:11px;color:var(--text-dim);margin-bottom:4px;}
.stat-box .value{font-size:17px;font-weight:700;}
.dist-row{display:flex;align-items:center;gap:10px;margin-bottom:8px;font-size:12px;}
.dist-name{width:56px;color:var(--text-dim);flex-shrink:0;}
.dist-bar-track{flex:1;height:8px;background:var(--bg);border-radius:4px;overflow:hidden;}
.dist-bar-fill{height:100%;border-radius:4px;transition:width .3s;}
.dist-pct{width:34px;text-align:right;color:var(--text-dim);flex-shrink:0;}
#donut-wrap{display:flex;align-items:center;gap:20px;margin-top:14px;}
#donut{width:120px;height:120px;border-radius:50%;flex-shrink:0;position:relative;}
#donut-center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;}
#donut-center b{font-size:16px;}
#donut-center span{font-size:10px;color:var(--text-dim);}
.legend{display:flex;flex-direction:column;gap:6px;font-size:12px;}
.legend-item{display:flex;align-items:center;gap:7px;}
.legend-dot{width:8px;height:8px;border-radius:50%;}
#status-bar{display:flex;align-items:center;justify-content:space-between;padding:10px 24px;border-top:1px solid var(--panel-border);background:var(--panel);font-size:12px;color:var(--text-dim);}
</style>
</head>
<body>
<div id="sidebar">
  <div class="brand">
    <div class="brand-icon">SM</div>
    <div><div class="brand-name">Smart Meeting AI</div><div class="brand-sub">Real-time speaker insights</div></div>
  </div>
  <nav>
    <div class="nav-item active">Dashboard</div>
    <div class="nav-item">Live Meet</div>
    <div class="nav-item">Transcripts</div>
    <div class="nav-item">Analytics</div>
    <div class="nav-item">Settings</div>
  </nav>
  <div class="sidebar-footer">TalkNet-ASD active speaker detection + Whisper subtitles, running locally.</div>
</div>
<div id="main">
  <header>
    <div style="font-weight:600;">Live Meeting</div>
    <div class="header-right">
      <div class="status-pill"><span class="dot" id="connDot"></span><span id="connText">Connecting...</span></div>
      <div id="timer" style="font-size:12px;color:var(--text-dim);">00:00</div>
    </div>
  </header>
  <div id="content">
    <div class="row row-top">
      <div class="card">
        <div class="card-header"><div class="card-title">Live Meeting Feed</div><span id="peopleCount" style="font-size:12px;color:var(--text-dim);">0 people detected</span></div>
        <div id="video-wrap"><img src="/video_feed"><div class="corner-tag">LIVE</div></div>
      </div>
      <div class="card">
        <div class="card-header"><div class="card-title">Active Speaker</div><div class="badge-live"><span class="dot"></span>LIVE</div></div>
        <div id="speaker-photo"><img src="/video_feed"></div>
        <div id="speaker-name">Waiting for speaker...</div>
        <div id="speaker-status"><span>No one is speaking</span></div>
        <div class="stat-row"><span>Speaking Score<b id="speaker-score">0.00</b></span></div>
      </div>
      <div class="card">
        <div class="card-header"><div class="card-title">Participants (<span id="pCount">0</span>)</div></div>
        <div id="participant-list"></div>
      </div>
    </div>
    <div class="row row-bottom">
      <div class="card">
        <div class="card-header"><div class="card-title">Live Transcript</div><div class="badge-live"><span class="dot"></span>Real-time</div></div>
        <div id="transcript"><div class="t-empty">No transcript yet</div></div>
      </div>
      <div class="card">
        <div class="card-header"><div class="card-title">Meeting Analytics</div></div>
        <div class="stat-grid">
          <div class="stat-box"><div class="label">Total Participants</div><div class="value" id="stat-total">0</div></div>
          <div class="stat-box"><div class="label">Active Speaker</div><div class="value" id="stat-active">-</div></div>
          <div class="stat-box"><div class="label">Speaker Changes</div><div class="value" id="stat-changes">0</div></div>
          <div class="stat-box"><div class="label">Meeting Duration</div><div class="value" id="stat-duration">00:00</div></div>
          <div class="stat-box"><div class="label">Avg. Speaking Time</div><div class="value" id="stat-avg">0s</div></div>
          <div class="stat-box"><div class="label">Transcript Segments</div><div class="value" id="stat-segments">0</div></div>
        </div>
        <div style="font-size:12px;font-weight:600;margin-bottom:8px;">Speaking Time Distribution</div>
        <div id="dist-list"></div>
        <div id="donut-wrap">
          <div id="donut"><div id="donut-center"><b id="donut-total">00:00</b><span>Total</span></div></div>
          <div class="legend" id="donut-legend"></div>
        </div>
      </div>
    </div>
  </div>
  <div id="status-bar"><div>Backend: /status, /subtitle_text, /transcript</div><div id="sysStatus">connecting...</div></div>
</div>
<script>
const COLORS=["#22c55e","#3b82f6","#a855f7","#f59e0b","#ef4444","#06b6d4"];
const startTime = Date.now();
const state = { speakMs:{}, lastStart:{}, colors:{}, order:[], currentSpeaker:null, changes:0, segments:0 };

function colorFor(id){
  if(!(id in state.colors)){
    state.colors[id] = COLORS[state.order.length % COLORS.length];
    state.order.push(id);
    state.speakMs[id]=0;
  }
  return state.colors[id];
}
function fmt(ms){const s=Math.floor(ms/1000);const m=Math.floor(s/60);const r=s%60;return String(m).padStart(2,"0")+":"+String(r).padStart(2,"0");}

async function pollStatus(){
  try{
    const r = await fetch("/status");
    const d = await r.json();
    document.getElementById("connDot").classList.remove("bad");
    document.getElementById("connText").innerText = "Connected";
    document.getElementById("sysStatus").innerText = "live";

    const attendees = Array.isArray(d.attendees) ? d.attendees : [];
    document.getElementById("peopleCount").innerText = d.persons_detected + " people detected";
    document.getElementById("pCount").innerText = attendees.length;
    document.getElementById("stat-total").innerText = attendees.length;

    attendees.forEach(function(a){ colorFor(a.id); });

    // active speaker
    if(d.active_speaker_id !== null && d.active_speaker_id !== undefined){
      const id = d.active_speaker_id;
      if(state.currentSpeaker !== id){ state.changes++; document.getElementById("stat-changes").innerText = state.changes; }
      state.currentSpeaker = id;
      if(!state.lastStart[id]) state.lastStart[id] = Date.now();
      document.getElementById("speaker-name").innerText = "Person " + id;
      document.getElementById("stat-active").innerText = "Person " + id;
      document.getElementById("speaker-status").innerHTML = "<span>Speaking</span>";
    } else {
      Object.keys(state.lastStart).forEach(function(id){
        if(state.lastStart[id]){ state.speakMs[id] = (state.speakMs[id]||0) + (Date.now()-state.lastStart[id]); state.lastStart[id]=null; }
      });
      state.currentSpeaker = null;
      document.getElementById("speaker-name").innerText = "Waiting for speaker...";
      document.getElementById("stat-active").innerText = "-";
      document.getElementById("speaker-status").innerHTML = "<span>No one is speaking</span>";
    }
    document.getElementById("speaker-score").innerText = (d.speaking_score||0).toFixed(2);

    // participants list
    const wrap = document.getElementById("participant-list");
    wrap.innerHTML = "";
    const totalMs = Object.values(state.speakMs).reduce(function(a,b){return a+b;},0) +
      Object.keys(state.lastStart).reduce(function(a,id){return a + (state.lastStart[id]?Date.now()-state.lastStart[id]:0);},0) || 1;
    attendees.forEach(function(a){
      const c = colorFor(a.id);
      const speaking = state.currentSpeaker === a.id;
      const ms = (state.speakMs[a.id]||0) + (state.lastStart[a.id]?Date.now()-state.lastStart[a.id]:0);
      const pct = Math.round((ms/totalMs)*100);
      const el = document.createElement("div");
      el.className = "participant";
      el.innerHTML = '<div class="avatar" style="background:'+c+'22;color:'+c+'">P'+a.id+'</div>'+
        '<div><div class="p-name">Person '+a.id+'</div><div class="p-status '+(speaking?"speaking":"")+'"><span class="dot"></span>'+(speaking?"Speaking":"Silent")+'</div></div>'+
        '<div class="p-pct">'+pct+'%</div>';
      wrap.appendChild(el);
    });

    // distribution + donut
    const distWrap = document.getElementById("dist-list");
    const legendWrap = document.getElementById("donut-legend");
    distWrap.innerHTML = ""; legendWrap.innerHTML = "";
    let grad = []; let acc = 0;
    const sorted = attendees.slice().sort(function(a,b){
      const ma = (state.speakMs[a.id]||0) + (state.lastStart[a.id]?Date.now()-state.lastStart[a.id]:0);
      const mb = (state.speakMs[b.id]||0) + (state.lastStart[b.id]?Date.now()-state.lastStart[b.id]:0);
      return mb-ma;
    });
    sorted.forEach(function(a){
      const c = colorFor(a.id);
      const ms = (state.speakMs[a.id]||0) + (state.lastStart[a.id]?Date.now()-state.lastStart[a.id]:0);
      const pct = totalMs ? (ms/totalMs)*100 : 0;
      const row = document.createElement("div");
      row.className = "dist-row";
      row.innerHTML = '<div class="dist-name">Person '+a.id+'</div><div class="dist-bar-track"><div class="dist-bar-fill" style="width:'+pct+'%;background:'+c+'"></div></div><div class="dist-pct">'+Math.round(pct)+'%</div>';
      distWrap.appendChild(row);
      const lg = document.createElement("div");
      lg.className = "legend-item";
      lg.innerHTML = '<span class="legend-dot" style="background:'+c+'"></span>Person '+a.id;
      legendWrap.appendChild(lg);
      grad.push(c+" "+acc+"% "+(acc+pct)+"%");
      acc += pct;
    });
    if(grad.length) document.getElementById("donut").style.background = "conic-gradient("+grad.join(",")+")";
    document.getElementById("stat-avg").innerText = attendees.length ? fmt(totalMs/attendees.length) : "0s";
  } catch(e){
    document.getElementById("connDot").classList.add("bad");
    document.getElementById("connText").innerText = "Disconnected";
    document.getElementById("sysStatus").innerText = "offline";
  }
}

async function pollTranscript(){
  try{
    const r = await fetch("/transcript");
    const d = await r.json();
    const list = Array.isArray(d) ? d : [];
    document.getElementById("stat-segments").innerText = list.length;
    const wrap = document.getElementById("transcript");
    if(list.length === 0){ wrap.innerHTML = '<div class="t-empty">No transcript yet</div>'; return; }
    wrap.innerHTML = "";
    list.slice(-30).forEach(function(item){
      const speaker = item.speaker || "Unknown";
      const idMatch = speaker.match(/\d+/);
      const c = idMatch ? colorFor(parseInt(idMatch[0])) : "#8b95a8";
      const line = document.createElement("div");
      line.className = "t-line";
      line.innerHTML = '<div class="t-time">'+(item.time||"")+'</div><div class="t-body"><b style="color:'+c+'">'+speaker+'</b><span>'+item.text+'</span></div>';
      wrap.appendChild(line);
    });
    wrap.scrollTop = wrap.scrollHeight;
  } catch(e){}
}

function tick(){
  document.getElementById("timer").innerText = fmt(Date.now()-startTime);
  document.getElementById("stat-duration").innerText = fmt(Date.now()-startTime);
  document.getElementById("donut-total").innerText = fmt(Date.now()-startTime);
}

setInterval(pollStatus, 700);
setInterval(pollTranscript, 1500);
setInterval(tick, 1000);
pollStatus();
pollTranscript();
</script>
</body>
</html>
"""


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
    audio_stream = sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS, dtype="float32", callback=audio_callback, blocksize=AUDIO_BLOCKSIZE)
    audio_stream.start()
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
        app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, threaded=True, use_reloader=False)
    finally:
        try:
            audio_stream.stop()
            audio_stream.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
