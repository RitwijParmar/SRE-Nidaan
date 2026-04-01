#!/usr/bin/env python3
"""
Add AI voiceover narration to the recorded SRE Nidaan demo video.

Input:
  presentations/SRE_Nidaan_Demo_Recording.webm
  presentations/voiceover_script.txt

Output:
  presentations/SRE_Nidaan_Demo_Recording_Voiceover_Indian.mp4
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

from moviepy import AudioFileClip, ImageClip, VideoFileClip, concatenate_videoclips


ROOT = Path(__file__).resolve().parents[1]
PRESENTATIONS = ROOT / "presentations"
INPUT_VIDEO = PRESENTATIONS / "SRE_Nidaan_Demo_Recording.webm"
SCRIPT_PATH = PRESENTATIONS / "voiceover_script.txt"
VOICE_AUDIO_AIFF = PRESENTATIONS / "SRE_Nidaan_Demo_Voiceover.aiff"
VOICE_AUDIO_M4A = PRESENTATIONS / "SRE_Nidaan_Demo_Recording_Voiceover_Indian.m4a"
VOICE_AUDIO_MP3 = PRESENTATIONS / "SRE_Nidaan_Demo_Recording_Voiceover_Indian.mp3"
OUTPUT_VIDEO = PRESENTATIONS / "SRE_Nidaan_Demo_Recording_Voiceover_Indian.mp4"


def run_say_tts(script_text: str, voice: str, rate: int) -> Path:
    cmd = ["say", "-v", voice, "-r", str(rate), "-o", str(VOICE_AUDIO_AIFF), script_text]
    subprocess.run(cmd, check=True)
    cmd = [
        "afconvert",
        "-f",
        "m4af",
        "-d",
        "aac",
        str(VOICE_AUDIO_AIFF),
        str(VOICE_AUDIO_M4A),
    ]
    subprocess.run(cmd, check=True)
    return VOICE_AUDIO_M4A


async def run_edge_tts(
    script_text: str,
    voice: str = "en-IN-PrabhatNeural",
    rate: str = "-4%",
    pitch: str = "-1Hz",
) -> Path:
    import edge_tts

    communicate = edge_tts.Communicate(script_text, voice=voice, rate=rate, pitch=pitch)
    await communicate.save(str(VOICE_AUDIO_MP3))
    return VOICE_AUDIO_MP3


def merge_audio_video(audio_path: Path) -> None:
    video = VideoFileClip(str(INPUT_VIDEO))
    audio = AudioFileClip(str(audio_path))

    target_video = video
    if audio.duration > video.duration:
        freeze_seconds = audio.duration - video.duration
        last_frame = video.get_frame(max(video.duration - 0.05, 0))
        freeze = ImageClip(last_frame).with_duration(freeze_seconds).with_fps(video.fps or 24)
        target_video = concatenate_videoclips([video, freeze], method="compose")
    elif video.duration > audio.duration:
        target_video = video.subclipped(0, audio.duration)

    final = target_video.with_audio(audio)
    final.write_videofile(
        str(OUTPUT_VIDEO),
        codec="libx264",
        audio_codec="aac",
        fps=video.fps or 24,
        bitrate="3500k",
    )

    final.close()
    target_video.close()
    video.close()
    audio.close()


def main() -> None:
    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(f"Missing input video: {INPUT_VIDEO}")
    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Missing script file: {SCRIPT_PATH}")

    script_text = SCRIPT_PATH.read_text(encoding="utf-8").strip()
    if not script_text:
        raise ValueError("Voiceover script is empty.")

    voice_engine = os.environ.get("VOICE_ENGINE", "edge").strip().lower()
    audio_path: Path

    if voice_engine == "say":
        voice = os.environ.get("MAC_TTS_VOICE", "Aman")
        rate = int(os.environ.get("MAC_TTS_RATE", "158"))
        audio_path = run_say_tts(script_text=script_text, voice=voice, rate=rate)
    else:
        voice = os.environ.get("EDGE_TTS_VOICE", "en-IN-PrabhatNeural")
        rate = os.environ.get("EDGE_TTS_RATE", "-4%")
        pitch = os.environ.get("EDGE_TTS_PITCH", "-1Hz")
        audio_path = asyncio.run(
            run_edge_tts(script_text=script_text, voice=voice, rate=rate, pitch=pitch)
        )

    merge_audio_video(audio_path=audio_path)
    print(f"Created narrated video: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
