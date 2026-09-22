REPLACEMENT FILES FOR SMART MEETING

Copy these files into the project root:

app.py
smart_meeting/config.py
smart_meeting/talknet_cpu.py
smart_meeting/speaker_engine.py
smart_meeting/subtitle_engine.py

Main fix:
- 25 TalkNet visual frames at 10 FPS = 2.5 seconds.
- MFCC window/step now scale with TARGET_FPS like the original TalkNet dataLoader.
- TalkNet receives a dedicated 2.5-second audio window.
- Whisper keeps its separate 3-second audio context.
- Removed circular/wrap MFCC padding; only the final MFCC frame is repeated if the fixed 4:1 TalkNet feature count needs a small pad.
- Speaker face window uses the real configured window duration.
- Warm-up requires at least 10 face crops.

After replacing files, delete all __pycache__ folders and run:
python -m py_compile app.py smart_meeting\\config.py smart_meeting\\talknet_cpu.py smart_meeting\\speaker_engine.py smart_meeting\\subtitle_engine.py
python app.py
