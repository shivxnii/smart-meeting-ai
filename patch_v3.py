"""
patch_v3.py - stops ONE person from getting a new ID again and again.

How to use:
  1. Put this file in the folder that CONTAINS the 'smart_meeting' folder
     (the same folder where app.py is).
  2. Open PowerShell in that folder and run:   python patch_v3.py
  3. It prints PATCH OK when done. Your old file is saved as
     smart_meeting/speaker_engine.py.bak

Safe: if anything does not match, it changes NOTHING and tells you.
"""
import os
import shutil
import sys

P = os.path.join("smart_meeting", "speaker_engine.py")

if not os.path.exists(P):
    print("ERROR: could not find", P)
    print("Run this from the folder that contains 'smart_meeting'.")
    print("Current folder:", os.getcwd())
    sys.exit(1)

s = open(P, encoding="utf-8-sig").read()

if "MAX_PEOPLE" in s:
    print("Already patched. Nothing to do.")
    sys.exit(0)


def rep(old, new):
    global s
    n = s.count(old)
    if n != 1:
        print("PATCH FAILED: expected to find this block once, found", n, "times:")
        print(old.strip().splitlines()[0])
        print("Nothing was changed. Send me your speaker_engine.py file.")
        sys.exit(1)
    s = s.replace(old, new)


rep("""#   4. Newly created tracks are no longer marked "missed" instantly.
""", """#   4. Newly created tracks are no longer marked "missed" instantly.
#   5. VERSION 3: MAX_PEOPLE cap, stale-track re-acquire, [ID] log lines.
#      (If you can read this line, you have the latest file.)
""")

rep("""# A track survives at least this many missed frames (~10 fps => 4 sec).
""", """# Hard cap on simultaneous people. 0 = unlimited (normal meeting).
# Set MAX_PEOPLE = 1 in config.py when only one person is in front of
# the camera: then that person can never receive a second ID.
MAX_PEOPLE = int(getattr(_cfg, "MAX_PEOPLE", 0))
# A visible track counts as "briefly lost" after this many missed frames.
STALE_MIN_MISSED = int(getattr(_cfg, "STALE_MIN_MISSED", 3))
# A track survives at least this many missed frames (~10 fps => 4 sec).
""")

rep("""        if (
            LOST_SOLO_REUSE
            and not self.tracks
            and len(self.lost_tracks) == 1
        ):
            return self.lost_tracks.pop().id

""", """        if (
            LOST_SOLO_REUSE
            and not self.tracks
            and (len(self.lost_tracks) == 1 or MAX_PEOPLE == 1)
        ):
            newest = max(self.lost_tracks, key=lambda t: t.last_seen)
            self.lost_tracks.remove(newest)
            return newest.id

""")

rep("""                if already_near_track:
                    continue

                crop = self._extract_face_crop(frame_bgr, box)

                revived_id = self._recover_lost_id(box, timestamp)

                track = PersonTrack(box, track_id=revived_id)

                track.update(box, crop, timestamp)

                self.tracks.append(track)

""", """                if already_near_track:
                    continue

                crop = self._extract_face_crop(frame_bgr, box)

                # ----------------------------------------------
                # Same person or new person?
                #
                # A visible track that was not matched this frame
                # (face temporarily missed) is the most likely owner
                # of a new, unmatched face. Hand the face to it instead
                # of creating a new ID.
                #   - Normal mode: only if it is close enough.
                #   - MAX_PEOPLE cap reached: always (a new face cannot
                #     be a new person), or ignore it as a false hit.
                # ----------------------------------------------
                capped = bool(MAX_PEOPLE) and len(self.tracks) >= MAX_PEOPLE

                min_missed = 1 if capped else STALE_MIN_MISSED

                stale = [
                    i for i, t in enumerate(self.tracks)
                    if i not in matched_tracks
                    and t.missed_frames >= min_missed
                ]

                pick = None

                if stale:

                    if capped:
                        pick = max(
                            stale,
                            key=lambda k: self.tracks[k].last_seen,
                        )
                    else:
                        best_d = LOST_MATCH_MAX_DIST
                        for i in stale:
                            t = self.tracks[i]
                            dd = (
                                center_distance(t.smooth_box, box)
                                / max(t.size, 1.0)
                            )
                            if dd <= best_d:
                                best_d = dd
                                pick = i

                if pick is not None:
                    t = self.tracks[pick]
                    t.smooth_box = list(box)
                    t.update(box, crop, timestamp)
                    matched_tracks.add(pick)
                    matched_boxes.add(box_index)
                    print(
                        f"[ID] Person {t.id} re-acquired "
                        "(same person, face was briefly lost)"
                    )
                    continue

                if capped:
                    # Cap reached and nobody to hand the face to:
                    # treat it as a false detection.
                    continue

                revived_id = self._recover_lost_id(box, timestamp)

                track = PersonTrack(box, track_id=revived_id)

                track.update(box, crop, timestamp)

                self.tracks.append(track)

                print(
                    f"[ID] Person {track.id} "
                    f"{'REVIVED (old id reused)' if revived_id is not None else 'NEW'}"
                    f" | visible={len(self.tracks)} lost={len(self.lost_tracks)}"
                )

""")

# Syntax check BEFORE touching the real file.
try:
    compile(s, P, "exec")
except SyntaxError as e:
    print("PATCH FAILED: result would not be valid Python:", e)
    print("Nothing was changed. Send me your speaker_engine.py file.")
    sys.exit(1)

shutil.copy(P, P + ".bak")
with open(P, "w", encoding="utf-8") as f:
    f.write(s)

print("PATCH OK. Backup saved as speaker_engine.py.bak")
print("Next: open smart_meeting/config.py and add  MAX_PEOPLE = 1  if only one person sits in front of the camera.")