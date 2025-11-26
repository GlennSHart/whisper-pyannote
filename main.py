import whisper
import torch
import torchaudio
import warnings
import time
import json
import os
from pyannote.audio import Pipeline

import tempfile
import subprocess

from tqdm import tqdm


def convert_to_wav(input_path):
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", "highpass=f=200, lowpass=f=3000, volume=1.5",
        "-ar", "16000", "-ac", "1",
        temp_wav
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return temp_wav


def transcribe_segment(model, wav_path, device, start, end):
    import tempfile, subprocess
    temp_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run([
        "ffmpeg", "-y", "-i", wav_path,
        "-ss", str(start), "-to", str(end),
        "-ar", "16000", "-ac", "1",
        temp_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    options = {"fp16": False} if device=="cpu" else {}
    result = model.transcribe(temp_path, **options, language="ru")
    os.remove(temp_path)
    return result["text"]


def diarize_audio(audio_path):
    pipeline = Pipeline.from_pretrained(
        'pyannote/speaker-diarization-community-1',
        token="" #insert token
    )

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    diarization = pipeline(audio_path)

    segments = []
    for turn, speaker in diarization.speaker_diarization:
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": f"speaker_{speaker}"
        })
    return segments


def transcribe_by_speaker(audio_path, model_name="tiny"): #change model if you need
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)

    wav_path = convert_to_wav(audio_path)
    
    segments = diarize_audio(wav_path)
    result_segments = []
    
    for seg in tqdm(segments, desc="Transcribing"):
        text = transcribe_segment(model, wav_path, device, seg["start"], seg["end"])
        result_segments.append({
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "text": text
        })

    merged_segments = merge_segments(result_segments)

    import os
    os.remove(wav_path)
    return merged_segments


def merge_segments(segments, gap_threshold=0.5):
    merged = []
    for seg in segments:
        if merged and seg["speaker"] == merged[-1]["speaker"] and seg["start"] - merged[-1]["end"] < gap_threshold:
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] += " " + seg["text"]
        else:
            merged.append(seg)
    return merged


def main(audio_file, output_json="transcription.json", model_name="medium"):
    warnings.filterwarnings("ignore", category=FutureWarning)
    start_time = time.time()
    segments = transcribe_by_speaker(audio_file, model_name)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"Saved transcription with {len(segments)} segments to {output_json}")
    print(f"Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    audio_file = "test.wav"
    main(audio_file)