#!/usr/bin/env python3
"""
Add AI voiceover narration to the recorded SRE Nidaan demo video.

Input:
  presentations/SRE_Nidaan_Demo_Recording.webm
  presentations/voiceover_script.txt

Output:
  presentations/SRE_Nidaan_Demo_Recording_Voiceover.mp4
  presentations/SRE_Nidaan_Demo_Recording_Voiceover.m4a
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from moviepy import AudioFileClip, ImageClip, VideoFileClip, concatenate_videoclips


ROOT = Path(__file__).resolve().parents[1]
PRESENTATIONS = ROOT / "presentations"
INPUT_VIDEO = PRESENTATIONS / "SRE_Nidaan_Demo_Recording.webm"
SCRIPT_PATH = PRESENTATIONS / "voiceover_script.txt"
VOICE_AUDIO_AIFF = PRESENTATIONS / "SRE_Nidaan_Demo_Voiceover.aiff"
VOICE_AUDIO_M4A = PRESENTATIONS / "SRE_Nidaan_Demo_Recording_Voiceover.m4a"
OUTPUT_VIDEO = PRESENTATIONS / "SRE_Nidaan_Demo_Recording_Voiceover.mp4"


def run_say_tts(script_text: str, voice: str = "Samantha", rate: int = 165) -> None:
    cmd = ["say", "-v", voice, "-r", str(rate), "-o", str(VOICE_AUDIO_AIFF), script_text]
    subprocess.run(cmd, check=True)


def convert_audio_to_m4a() -> None:
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


def merge_audio_video() -> None:
    video = VideoFileClip(str(INPUT_VIDEO))
    audio = AudioFileClip(str(VOICE_AUDIO_M4A))

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

    run_say_tts(script_text=script_text)
    convert_audio_to_m4a()
    merge_audio_video()
    print(f"Created narrated video: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()

