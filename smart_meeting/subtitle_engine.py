import time
import threading
import re

import cv2
import numpy as np
import whisper

from smart_meeting.config import (
    WHISPER_LANGUAGE,
    WHISPER_TASK,
    WHISPER_AUDIO_RMS_THRESHOLD,
)


class SubtitleEngine:
    """
    Multilingual subtitle engine.

    Supports:
    - Hindi
    - English
    - Hindi-English Hinglish
    - Automatic language detection
    - Translates everything to English subtitles
    """

    def __init__(self, model_name="base"):
        self.model_name = model_name
        self.lock = threading.Lock()

        print(
            f"[Whisper] Loading multilingual model '{model_name}'..."
        )

        self.model = whisper.load_model(
            model_name,
            device="cpu",
        )

        print("[Whisper] Multilingual Whisper model loaded.")

        self.last_text = ""
        self.last_speaker = ""
        self.last_transcription_time = 0.0
        self.history = []
        self.max_history = 300

    def get_history(self):
        with self.lock:
            return list(self.history)

    def _record_history(self, speaker_name, text):
        with self.lock:
            self.history.append({
                "speaker": speaker_name,
                "text": text,
                "time": time.strftime("%H:%M:%S"),
            })
            if len(self.history) > self.max_history:
                self.history.pop(0)

    # ========================================================
    # AUDIO PREPARATION
    # ========================================================

    def _prepare_audio(self, audio, sample_rate):
        if audio is None:
            return None

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        if audio.size == 0:
            return None

        # Convert stereo/multichannel audio tomono.
        if audio.ndim > 1:
            audio = np.mean(
                audio,
                axis=1,
            )

        audio = np.nan_to_num(
            audio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # Remove DC offset.
        audio = audio - np.mean(audio)

        # Normalize only when required.
        peak = float(
            np.max(
                np.abs(audio)
            )
        )

        if peak > 1.0:
            audio = audio / peak

        # Resample to Whisper's required 16 kHz.
        if sample_rate != 16000:
            target_length = int(
                len(audio) * 16000 / sample_rate
            )

            if target_length <= 0:
                return None

            audio = np.interp(
                np.linspace(
                    0,
                    len(audio) - 1,
                    target_length,
                ),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)

        # Gentle peak normalization.
        peak = float(
            np.max(
                np.abs(audio)
            )
        )

        if peak > 0.0:
            audio = audio / max(peak, 1.0)

        return audio.astype(np.float32)

    # ========================================================
    # TEXT CLEANING
    # ========================================================

    @staticmethod
    def _clean_text(text):
        if not text:
            return ""

        text = text.strip()

        # Remove repeated whitespace.
        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        # Remove accidental leading/trailing punctuation.
        text = text.strip(
            " \t\n\r.,;:!?à¥¤"
        )

        return text

    @staticmethod
    def _looks_like_hallucination(
        text,
        duration_seconds=None,
    ):
        if not text:
            return True

        words = text.split()

        if len(words) > 55:
            return True

        if duration_seconds:
            words_per_second = (
                len(words) / max(duration_seconds, 0.1)
            )

            # Very high speech rate usually means hallucination.
            if words_per_second > 5.5:
                return True

        lowered_text = text.lower()

        # Detect the model echoing back our own prompt text
        # instead of transcribing real speech.
        prompt_echo_phrases = [
            "translate the exact spoken words",
            "transcribe the exact spoken words",
            "clear, accurate english",
            "do not invent words",
            "hindi, english, or hinglish",
        ]

        for phrase in prompt_echo_phrases:
            if phrase in lowered_text:
                return True

        # Common Whisper hallucinations that appear on
        # silence / near-silence / background noise.
        # These come from Whisper's training data (YouTube
        # captions, lectures, etc.) and are NOT real speech.
        known_hallucinations = [
            "i will start with the first one",
            "thanks for watching",
            "thank you for watching",
            "please subscribe",
            "subscribe to my channel",
            "see you in the next video",
            "let's get started",
            "let's begin",
            "okay so",
            "so today we are going to",
            "welcome back to my channel",
            "don't forget to like and subscribe",
            "thank you.",
            "bye bye",
            "goodbye",
            "the end",
            "music playing",
            "applause",
        ]

        for phrase in known_hallucinations:
            if phrase in lowered_text:
                return True

        if len(words) < 8:
            return False

        lowered = [
            word.lower()
            for word in words
        ]

        # Detect repeated phrase loops.
        for phrase_length in (2, 3, 4, 5):
            if len(words) < phrase_length * 3:
                continue

            phrases = []

            for index in range(
                len(words) - phrase_length + 1
            ):
                phrase = " ".join(
                    lowered[
                        index:index + phrase_length
                    ]
                )

                phrases.append(phrase)

            counts = {}

            for phrase in phrases:
                counts[phrase] = (
                    counts.get(phrase, 0) + 1
                )

            if counts:
                maximum_repeat = max(
                    counts.values()
                )

                threshold = (
                    5
                    if phrase_length == 2
                    else 4
                )

                if maximum_repeat >= threshold:
                    return True

        return False

    # ========================================================
    # TRANSCRIPTION
    # ========================================================

    def transcribe_audio_segment(
        self,
        audio,
        sample_rate=16000,
        speaker_name="Unknown Speaker",
    ):
        prepared_audio = self._prepare_audio(
            audio,
            sample_rate,
        )

        if prepared_audio is None:
            return ""

        duration_seconds = (
            len(prepared_audio) / 16000.0
        )

        if duration_seconds < 1.0:
            return ""

        # RMS-based silence detection.
        # Threshold raised to reject quiet room noise that
        # was previously slipping through and causing
        # Whisper to hallucinate text.
        rms = float(
            np.sqrt(
                np.mean(
                    prepared_audio ** 2
                )
            )
        )

        effective_rms_threshold = max(
            WHISPER_AUDIO_RMS_THRESHOLD,
            0.006,
        )

        if rms < effective_rms_threshold:
            return ""

        try:
            with self.lock:
                result = self.model.transcribe(
                    prepared_audio,

                    # Hindi, English, ya Hinglish automatic detect hoga.
                    language=None,

                    # Kisi bhi language mein bola gaya audio,
                    # English text mein translate hoga.
                    task="translate",

                    fp16=False,
                    temperature=0.0,

                    condition_on_previous_text=False,

                    # Stricter — reduces hallucination on
                    # silence/background noise.
                    no_speech_threshold=0.7,
                    logprob_threshold=-1.0,
                    compression_ratio_threshold=2.2,

                    # Short, neutral vocabulary hint only —
                    # NOT an instruction, so Whisper won't echo it back.
                    initial_prompt="Hindi English Hinglish meeting.",
                )
        except Exception as exc:
            print(
                "[Whisper] Transcription error:",
                repr(exc),
            )

            return ""

        text = ""

        if isinstance(result, dict):
            text = result.get(
                "text",
                "",
            )

        text = self._clean_text(text)

        if not text:
            return ""

        if self._looks_like_hallucination(
            text,
            duration_seconds,
        ):
            print(
                "[Whisper] Ignored possible hallucination:",
                text,
            )

            return ""

        # Avoid repeating the same subtitle continuously.
        with self.lock:
            if (
                text == self.last_text
                and speaker_name == self.last_speaker
            ):
                return (
                    f"{speaker_name}: {text}"
                )

            self.last_text = text
            self.last_speaker = speaker_name
            self.last_transcription_time = time.time()

        self._record_history(speaker_name, text)

        return (
            f"{speaker_name}: {text}"
        )

    # ========================================================
    # SUBTITLE OVERLAY
    # ========================================================

    @staticmethod
    def draw_subtitles_overlay(
        frame,
        subtitle_text="",
    ):
        if (
            frame is None
            or not subtitle_text
        ):
            return frame

        output = frame.copy()

        height, width = output.shape[:2]

        words = subtitle_text.split()

        lines = []
        current_line = ""

        # Smaller width prevents subtitles from going outside frame.
        max_chars = 48

        for word in words:
            candidate = (
                f"{current_line} {word}"
            ).strip()

            if len(candidate) <= max_chars:
                current_line = candidate
            else:
                if current_line:
                    lines.append(current_line)

                current_line = word

        if current_line:
            lines.append(current_line)

        lines = lines[-3:]

        if not lines:
            return output

        line_height = 34
        padding = 14

        box_height = (
            len(lines) * line_height
            + padding * 2
        )

        box_y1 = max(
            0,
            height - box_height - 18,
        )

        box_y2 = min(
            height,
            height - 18,
        )

        overlay = output.copy()

        cv2.rectangle(
            overlay,
            (12, box_y1),
            (width - 12, box_y2),
            (0, 0, 0),
            -1,
        )

        output = cv2.addWeighted(
            overlay,
            0.72,
            output,
            0.28,
            0,
        )

        for index, line in enumerate(lines):
            y = (
                box_y1
                + padding
                + (index + 1) * line_height
                - 5
            )

            cv2.putText(
                output,
                line,
                (25, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return output
