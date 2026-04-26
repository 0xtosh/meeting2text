#!/usr/bin/env python3
"""
meeting2text.py
───────────────────────────────────────────────────────────────────────────────────────────────────
Converts a meeting recording to a clean transcript and (in cloud mode) a
structured summary. Outputs a transcript .txt file and in cloud mode a _summary.txt output file 
───────────────────────────────────────────────────────────────────────────────────────────────────
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests


# ── NVIDIA NIM ────────────────────────────────────────────────────────────────
NIM_BASE_URL     = "https://integrate.api.nvidia.com/v1"

# Primary Models
TRANSCRIBE_MODEL = "microsoft/phi-4-multimodal-instruct"
SUMMARY_MODEL    = "mistralai/mistral-large-3-675b-instruct-2512"

# Fallback Chains for Resiliency (Used if primary is DEGRADED)
ASR_FALLBACKS = [
    "microsoft/phi-4-multimodal-instruct",
    "openai/whisper-large-v3",
    "nvidia/parakeet-ctc-1.1b-asr"
]

SUMMARY_FALLBACKS = [
    "mistralai/mistral-large-3-675b-instruct-2512",
    "meta/llama-3.1-405b-instruct",
    "deepseek-ai/deepseek-v3.2"
]

# ── DEFAULTS ──────────────────────────────────────────────────────────────────
FFMPEG_DEFAULT        = "ffmpeg"
CHUNK_MINUTES_DEFAULT = 5           
WHISPER_MODEL_DEFAULT = "medium.en" # You will need about 1,5 GB of diskspace for the model

# ── OPTIONAL: hard-code your NVIDIA API key here (cloud mode only) ────────────
NVIDIA_API_KEY_DEFAULT = ""


# ─────────────────────────────────────────────────────────────────────────────
# TIMER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fmt_t(seconds: float) -> str:
    """Format a duration in seconds as a compact human-readable string."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m {s % 60:02d}s"


def phase_line(label: str, phase_secs: float, total_secs: float):
    """Print a tidy one-line timing summary for a completed phase."""
    print(f"      phase {fmt_t(phase_secs):>7}  |  total {fmt_t(total_secs):>7}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe and optionally summarise a meeting recording.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("input", help="Path to the input file.")
    parser.add_argument("--mode", required=True, choices=["local", "cloud"], help="Transcription backend.")
    parser.add_argument("--ffmpeg", default=FFMPEG_DEFAULT, help="Path to ffmpeg executable.")
    parser.add_argument("--output", default=None, help="Transcript output path.")
    parser.add_argument("--keep-wav", action="store_true", help="Keep intermediate WAV.")

    local_grp = parser.add_argument_group("local mode options")
    local_grp.add_argument("--whisper-model", default=WHISPER_MODEL_DEFAULT, help="Whisper model size.")

    cloud_grp = parser.add_argument_group("cloud mode options")
    cloud_grp.add_argument("--api-key", default=None, help="NVIDIA NIM API key.")
    cloud_grp.add_argument("--chunk-min", type=float, default=CHUNK_MINUTES_DEFAULT, help="Chunk length in minutes.")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# FFMPEG HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def run_cmd(cmd: list[str], step_label: str) -> subprocess.CompletedProcess:
    """Run a subprocess command; exit cleanly on failure."""
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"\n  ERROR during {step_label} (exit {result.returncode}):")
        print("  " + "─" * 60)
        print(result.stderr[-2000:])
        print("  " + "─" * 60)
        sys.exit(1)
    return result


def extract_audio(ffmpeg: str, src: Path, wav: Path, t_start: float) -> float:
    """Extract 16-kHz mono PCM WAV from any media file."""
    t_phase_start = time.time()
    print(f"\n[1/4] Extracting audio -> {wav.name}")
    cmd = [
        ffmpeg, "-y", "-i", str(src), "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(wav),
    ]
    run_cmd(cmd, "audio extraction")
    t_end = time.time()
    size_mb = wav.stat().st_size / 1_048_576
    print(f"      cmd: {' '.join(cmd)}")
    print(f"      WAV size : {size_mb:.1f} MB")
    print(f"      phase {fmt_t(t_end - t_phase_start):>7}  |  total {fmt_t(t_end - t_start):>7}")
    return t_end


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL MODE
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_local(wav_path: Path, txt_path: Path, model_size: str, t_start: float) -> float:
    from faster_whisper import WhisperModel
    from tqdm import tqdm

    print(f"\n[2/4] Loading Whisper model '{model_size}' (int8, CPU)...")
    t_load_start = time.time()
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    t_loaded = time.time()
    phase_line("Model load", t_loaded - t_load_start, t_loaded - t_start)

    print(f"\n[3/4] Analysing and transcribing...")
    t_phase_start = time.time()
    segments, info = model.transcribe(str(wav_path), beam_size=5, vad_filter=True)
    duration = info.duration

    with open(txt_path, "w", encoding="utf-8") as f, \
         tqdm(total=round(duration), unit="s", bar_format="{l_bar}{bar}| {n:.0f}s/{total:.0f}s", colour="green") as pbar:
        prev_end = 0.0
        for segment in segments:
            line = f"[{segment.start:.1f}s] {segment.text.strip()}"
            tqdm.write(f"  {line}")
            f.write(line + "\n")
            pbar.update(round(segment.end - prev_end))
            prev_end = segment.end

    t_end = time.time()
    phase_line("Transcription", t_end - t_phase_start, t_end - t_start)
    return t_end


def run_local(args, src: Path, wav: Path, txt: Path, t_start: float):
    print("=" * 62)
    print("  Meeting Transcriber  (local / faster-whisper edition)")
    print("=" * 62)
    extract_audio(args.ffmpeg, src, wav, t_start)
    transcribe_local(wav, txt, args.whisper_model, t_start)
    if not args.keep_wav and wav.exists(): wav.unlink()
    print(f"\nDONE! Total elapsed {fmt_t(time.time() - t_start)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLOUD MODE
# ─────────────────────────────────────────────────────────────────────────────

def split_wav(ffmpeg: str, wav: Path, chunk_seconds: float, tmp_dir: Path, t_start: float) -> tuple[list[Path], float]:
    t_phase_start = time.time()
    print(f"\n[2/4] Splitting audio into {chunk_seconds/60:.0f}-minute chunks...")
    pattern = str(tmp_dir / "chunk_%04d.wav")
    cmd = [
        ffmpeg, "-y", "-i", str(wav),
        "-f", "segment", "-segment_time", str(int(chunk_seconds)),
        "-acodec", "libmp3lame", "-b:a", "32k", "-ar", "16000", "-ac", "1",
        pattern,
    ]
    run_cmd(cmd, "audio splitting")
    chunks = sorted(tmp_dir.glob("chunk_*.wav"))
    t_end = time.time()
    print(f"      {len(chunks)} chunks created")
    print(f"      phase {fmt_t(t_end - t_phase_start):>7}  |  total {fmt_t(t_end - t_start):>7}")
    return chunks, t_end


def transcribe_chunk(api_key: str, chunk_path: Path, chunk_index: int, total: int, t_start: float) -> str:
    audio_b64 = base64.b64encode(chunk_path.read_bytes()).decode()
    t_chunk_start = time.time()

    for model_name in ASR_FALLBACKS:
        is_chat_api = "phi-4" in model_name
        url = f"{NIM_BASE_URL}/chat/completions" if is_chat_api else f"{NIM_BASE_URL}/audio/transcriptions"
        
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": [
                {"type": "audio_url", "audio_url": {"url": f"data:audio/mpeg;base64,{audio_b64}"}},
                {"type": "text", "text": "Transcribe every word spoken exactly. No commentary."}
            ]}],
            "max_tokens": 4096, "temperature": 0.0,
        } if is_chat_api else {"model": model_name, "content": audio_b64, "language": "en"}

        try:
            resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=payload, timeout=120)
            if resp.status_code == 400 and "DEGRADED" in resp.text: continue
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"] if is_chat_api else data.get("text", "")
            
            t_now = time.time()
            size_kb = chunk_path.stat().st_size / 1024
            print(f"      [{chunk_index}/{total}] {chunk_path.name} ({int(size_kb)} KB, {len(text)} chars)  phase {fmt_t(t_now - t_chunk_start):>7}  |  total {fmt_t(t_now - t_start):>7}")
            return text.strip()
        except Exception: continue

    return f"[TRANSCRIPTION FAILED FOR CHUNK {chunk_index}]"


def summarise_transcript(api_key: str, transcript: str, recording_name: str, t_start: float) -> tuple[str, float]:
    t_phase_start = time.time()
    
    system_prompt = (
        "You are an expert meeting note-taker. Produce a comprehensive summary in pure plain text. "
        "Do NOT use Markdown formatting (no asterisks, no bolding, no tables). Use ONLY the exact markers provided."
    )
    
    user_prompt = f"""Produce a comprehensive meeting summary in pure plain text for: "{recording_name}".
Use the exact headings shown (surrounded by === markers). Do not use Markdown.

=== SHORT DESCRIPTION ===
[Summary in 1-2 sentences]

=== LONG DESCRIPTION ===
[Detailed paragraph 4-8 sentences]

=== COMPLETE NARRATIVE ===
[300-600 words chronological account]

=== KEY TAKEAWAYS ===
[Bullet list using bullets like •]

=== THINGS AGREED ===
[Bullet list or "None recorded."]

=== ACTION POINTS ===
[Bullet list: • [Owner if known] - Description]

=== TOPICS FOR FOLLOW-UP ===
[Bullet list or "None recorded."]

=== NOTABLE QUOTES ===
[Bullet list: • [Quote text]]

TRANSCRIPT:
{transcript}
"""

    for model_name in SUMMARY_FALLBACKS:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 4096, "temperature": 0.3
        }
        try:
            resp = requests.post(f"{NIM_BASE_URL}/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=payload, timeout=180)
            if resp.status_code == 400 and "DEGRADED" in resp.text: continue
            resp.raise_for_status()
            t_end = time.time()
            print(f"      phase {fmt_t(t_end - t_phase_start):>7}  |  total {fmt_t(t_end - t_start):>7}")
            return resp.json()["choices"][0]["message"]["content"].strip(), t_end - t_phase_start
        except Exception: continue
    return "[SUMMARY FAILED ALL FALLBACKS]", time.time() - t_phase_start


def run_cloud(args, src: Path, wav: Path, txt: Path, t_start: float):
    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY", "") or NVIDIA_API_KEY_DEFAULT
    if not api_key: print("\n  ERROR: No NVIDIA API key found."); sys.exit(1)

    summary_file = txt.with_name(txt.stem + "_summary.txt")
    tmp_dir = src.parent / f"_chunks_{src.stem}"
    tmp_dir.mkdir(exist_ok=True)
    chunk_seconds = args.chunk_min * 60

    print()
    print("  CLOUD MODE -- DATA PRIVACY NOTICE")
    print("  " + "─" * 58)
    print("  Your audio will be uploaded to NVIDIA's servers and")
    print("  processed by the following third-party models:")
    print(f"    Transcription : {TRANSCRIBE_MODEL}")
    print(f"    Summary       : {SUMMARY_MODEL}")
    print(f"  Endpoint : {NIM_BASE_URL}")
    print(f"  File     : {src.name}")
    print()
    print("  Do not proceed if your recording contains confidential,")
    print("  sensitive, or personally identifiable information that")
    print("  must not leave your organisation's systems.")
    print("  " + "─" * 58)
    
    try:
        ans = input("  Type 'yes' to confirm and continue, anything else to abort: ").strip().lower()
    except (EOFError, KeyboardInterrupt): sys.exit(0)
    if ans != "yes": sys.exit(0)

    print()
    print("==============================================================")
    print("  Meeting Transcriber  (NVIDIA NIM edition)")
    print("==============================================================")
    print(f"  Input          : {src}")
    print(f"  Transcript out : {txt}")
    print(f"  Summary out    : {summary_file}")
    print(f"  Chunk size     : {args.chunk_min} min  ({int(chunk_seconds)} s)")
    print(f"  Transcribe     : {TRANSCRIBE_MODEL}")
    print(f"  Summarise      : {SUMMARY_MODEL}")
    print(f"  ffmpeg         : {args.ffmpeg}")
    print(f"  Started        : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("==============================================================")

    t_extract_done = extract_audio(args.ffmpeg, src, wav, t_start)
    chunks, t_split_done = split_wav(args.ffmpeg, wav, chunk_seconds, tmp_dir, t_start)

    parts = []
    print(f"\n[3/4] Transcribing {len(chunks)} chunks via {TRANSCRIBE_MODEL}...")
    print(f"      (Rate limit: 40 req/min - sleeping 1.5 s between calls)\n")
    t_trans_start = time.time()
    for i, chunk in enumerate(chunks, 1):
        parts.append(transcribe_chunk(api_key, chunk, i, len(chunks), t_start))
        chunk.unlink()
        if i < len(chunks): time.sleep(1.5)
    t_trans_end = time.time()
    print(f"      phase {fmt_t(t_trans_end - t_trans_start):>7}  |  total {fmt_t(t_trans_end - t_start):>7}")

    full_transcript = "\n\n".join(parts)
    txt.write_text(full_transcript, encoding="utf-8")
    print(f"\n      Transcript saved -> {txt}")
    print(f"      Total length     : {len(full_transcript):,} chars  /  ~{len(full_transcript.split()):,} words")

    print(f"\n[4/4] Generating structured summary with {SUMMARY_MODEL}...")
    print(f"      Transcript length : {len(full_transcript):,} chars  / ~{len(full_transcript.split()):,} words")
    summary_text, phase_sum = summarise_transcript(api_key, full_transcript, src.stem, t_start)
    
    header = (
        f"MEETING SUMMARY\n"
        f"{'─' * 62}\n"
        f"Recording  : {src.name}\n"
        f"Transcript : {txt.name}\n"
        f"Generated  : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Model      : {SUMMARY_MODEL}\n"
        f"{'─' * 62}\n\n"
    )
    summary_file.write_text(header + summary_text, encoding="utf-8")
    print(f"      Summary saved    -> {summary_file}")
    if not args.keep_wav:
        wav.unlink()
        print(f"      WAV deleted      -> {wav.name}")
    
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n" + "═" * 62)
    print(f"  Done!  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("─" * 62)
    print(f"  Phase timings:")
    print(f"    Audio extraction         {fmt_t(t_extract_done - t_start):>5}")
    print(f"    Audio splitting          {fmt_t(t_split_done - t_extract_done):>5}")
    print(f"    Transcription            {fmt_t(t_trans_end - t_trans_start):>5}  ({len(chunks)} chunks)")
    print(f"    Summary                  {fmt_t(phase_sum):>5}")
    print("─" * 62)
    print(f"  Total elapsed            {fmt_t(time.time() - t_start):>5}")
    print("─" * 62)
    print(f"  Outputs:")
    print(f"    {txt}")
    print(f"    {summary_file}")
    print("═" * 62 + "\n")


def main():
    args = parse_args()
    src = Path(args.input).resolve()
    if not src.exists(): sys.exit(1)
    wav = src.with_suffix(".wav")
    txt = Path(args.output).resolve() if args.output else src.with_suffix(".txt")

    if args.mode == "local": run_local(args, src, wav, txt, time.time())
    else: run_cloud(args, src, wav, txt, time.time())


if __name__ == "__main__":
    main()