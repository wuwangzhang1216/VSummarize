#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video -> Audio -> Transcription -> Cleaning -> Summary Pipeline
- Audio extraction: ffmpeg
- Speech recognition (ASR): OpenAI Audio Transcriptions
- Text cleaning & summarization: OpenAI Chat Completions

Dependencies:
  pip install openai python-dotenv tqdm
Environment:
  OPENAI_API_KEY=xxx
"""

import argparse
import json
import os
import subprocess
import sys
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI, APIError, RateLimitError, APITimeoutError


# ------------------ Utility Functions ------------------

def is_audio_file(file_path: Path) -> bool:
    """
    Check if the file is an audio file based on extension.
    """
    audio_extensions = {
        '.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma',
        '.opus', '.webm', '.oga', '.mp2', '.aiff', '.ape'
    }
    return file_path.suffix.lower() in audio_extensions


def is_video_file(file_path: Path) -> bool:
    """
    Check if the file is a video file based on extension.
    """
    video_extensions = {
        '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm',
        '.m4v', '.mpg', '.mpeg', '.3gp', '.f4v', '.ogv', '.ts'
    }
    return file_path.suffix.lower() in video_extensions


def convert_audio_to_wav(input_path: Path, output_path: Path, samplerate: int = 16000) -> None:
    """
    Convert any audio format to mono WAV using ffmpeg.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-acodec", "pcm_s16le",
        "-ac", "1",           # mono
        "-ar", str(samplerate),
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg and ensure it's in PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg audio conversion failed: {e.stderr.decode(errors='ignore')[:500]}")


def run_ffmpeg_extract_audio(video_path: Path, audio_path: Path, samplerate: int = 16000) -> None:
    """
    Extract mono WAV audio from video using ffmpeg.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",                # no video
        "-acodec", "pcm_s16le",
        "-ac", "1",           # mono
        "-ar", str(samplerate),
        str(audio_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg and ensure it's in PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg audio extraction failed: {e.stderr.decode(errors='ignore')[:500]}")


def backoff_retry(fn, *, retries: int = 5, base_delay: float = 1.5):
    """
    Exponential backoff retry wrapper for API calls.
    """
    for i in range(retries):
        try:
            return fn()
        except (RateLimitError, APITimeoutError, APIError) as e:
            if i == retries - 1:
                raise
            delay = base_delay * (2 ** i)
            print(f"API error, retrying in {delay:.1f}s...")
            time.sleep(delay)


def get_audio_duration_ffmpeg(audio_path: Path) -> float:
    """
    Get audio duration using ffmpeg probe.
    """
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except:
        # Fallback: assume a default duration
        return 0.0


def split_audio_to_chunks(audio_path: Path, max_size_mb: float = 24.0, chunk_duration_sec: float = 600.0) -> List[Path]:
    """
    Split audio into chunks under max_size_mb using ffmpeg.
    Returns list of chunk file paths.
    """
    max_size_bytes = max_size_mb * 1024 * 1024

    # Check if file is already small enough
    if audio_path.stat().st_size <= max_size_bytes:
        return [audio_path]

    # Get duration
    duration = get_audio_duration_ffmpeg(audio_path)
    if duration <= 0:
        return [audio_path]

    # Create temp directory for chunks
    temp_dir = Path(tempfile.mkdtemp(prefix="audio_chunks_"))
    chunk_paths = []

    # Calculate number of chunks needed
    num_chunks = max(1, int(duration / chunk_duration_sec) + 1)
    actual_chunk_duration = duration / num_chunks

    for i in range(num_chunks):
        start_time = i * actual_chunk_duration
        chunk_path = temp_dir / f"chunk_{i:03d}.mp3"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-ss", str(start_time),
            "-t", str(actual_chunk_duration),
            "-acodec", "mp3",
            "-b:a", "64k",  # Lower bitrate to ensure smaller file size
            "-ac", "1",      # Mono
            "-ar", "16000",  # Lower sample rate
            str(chunk_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            # Verify chunk size
            if chunk_path.exists() and chunk_path.stat().st_size <= max_size_bytes:
                chunk_paths.append(chunk_path)
            else:
                # If still too large, need smaller chunks
                print(f"Warning: Chunk {i} still too large, may need adjustment")
                chunk_paths.append(chunk_path)
        except subprocess.CalledProcessError as e:
            print(f"Error creating chunk {i}: {e}")
            continue

    return chunk_paths if chunk_paths else [audio_path]


def transcribe_single_chunk(client: OpenAI, chunk_info: Tuple[int, Path, float], asr_model: str, language: str) -> Tuple[int, Dict[str, Any], float]:
    """
    Transcribe a single audio chunk. Returns (index, transcription_data, accumulated_time).
    """
    idx, chunk_path, accumulated_time = chunk_info

    with chunk_path.open("rb") as f:
        def _call():
            return client.audio.transcriptions.create(
                model=asr_model,
                file=f,
                response_format="verbose_json",
                temperature=0.0,
                language=None if language == "auto" else language,
            )
        resp = backoff_retry(_call)

    chunk_data = resp if isinstance(resp, dict) else resp.model_dump()
    return idx, chunk_data, accumulated_time


def transcribe_audio(client: OpenAI, audio_path: Path, asr_model: str, language: str = "auto", parallel: bool = True, max_workers: int = 3) -> Dict[str, Any]:
    """
    Call OpenAI speech recognition API. Handles large files by splitting into chunks.
    """
    MAX_FILE_SIZE_MB = 24.0  # OpenAI limit is 25MB, use 24MB to be safe
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)

    # If file is small enough, transcribe directly
    if file_size_mb <= MAX_FILE_SIZE_MB:
        with audio_path.open("rb") as f:
            def _call():
                return client.audio.transcriptions.create(
                    model=asr_model,
                    file=f,
                    response_format="verbose_json",
                    temperature=0.0,
                    language=None if language == "auto" else language,
                )
            resp = backoff_retry(_call)
        data = resp if isinstance(resp, dict) else resp.model_dump()
        return data

    # File is too large, split into chunks
    print(f"  File size ({file_size_mb:.1f}MB) exceeds limit. Splitting into chunks...")
    chunks = split_audio_to_chunks(audio_path, max_size_mb=MAX_FILE_SIZE_MB)

    # Calculate accumulated times for each chunk
    chunk_infos = []
    accumulated_time = 0.0
    for i, chunk_path in enumerate(chunks):
        chunk_infos.append((i, chunk_path, accumulated_time))
        if i < len(chunks) - 1:
            chunk_duration = get_audio_duration_ffmpeg(chunk_path)
            accumulated_time += chunk_duration

    # Process chunks
    results = {}

    if parallel and len(chunks) > 1:
        print(f"  Processing {len(chunks)} chunks in parallel (max {max_workers} workers)...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all transcription tasks
            future_to_chunk = {
                executor.submit(transcribe_single_chunk, client, chunk_info, asr_model, language): chunk_info[0]
                for chunk_info in chunk_infos
            }

            # Process completed transcriptions
            with tqdm(total=len(chunks), desc="Transcribing chunks") as pbar:
                for future in as_completed(future_to_chunk):
                    idx = future_to_chunk[future]
                    try:
                        chunk_idx, chunk_data, chunk_accumulated_time = future.result()
                        results[chunk_idx] = (chunk_data, chunk_accumulated_time)
                        pbar.update(1)
                    except Exception as e:
                        print(f"\nError transcribing chunk {idx}: {e}")
                        results[idx] = ({"text": "", "segments": []}, 0)
                        pbar.update(1)
    else:
        # Sequential processing
        print(f"  Processing {len(chunks)} chunks sequentially...")
        for chunk_info in tqdm(chunk_infos, desc="Transcribing chunks"):
            idx, chunk_data, chunk_accumulated_time = transcribe_single_chunk(
                client, chunk_info, asr_model, language
            )
            results[idx] = (chunk_data, chunk_accumulated_time)

    # Combine results in order
    all_segments = []
    all_text = []

    for i in sorted(results.keys()):
        chunk_data, chunk_accumulated_time = results[i]

        # Collect text
        if "text" in chunk_data:
            all_text.append(chunk_data["text"])

        # Adjust segment timestamps and collect
        if "segments" in chunk_data:
            for seg in chunk_data["segments"]:
                adjusted_seg = seg.copy()
                adjusted_seg["start"] = seg.get("start", 0) + chunk_accumulated_time
                adjusted_seg["end"] = seg.get("end", 0) + chunk_accumulated_time
                all_segments.append(adjusted_seg)

    # Clean up temporary chunks if they were created
    if len(chunks) > 1:
        for chunk_path in chunks:
            try:
                chunk_path.unlink()
            except:
                pass
        # Remove temp directory
        try:
            chunks[0].parent.rmdir()
        except:
            pass

    # Combine results
    combined_data = {
        "text": " ".join(all_text),
        "segments": all_segments,
        "language": chunk_data.get("language", ""),
        "duration": accumulated_time if accumulated_time > 0 else chunk_data.get("duration", 0)
    }

    return combined_data


def save_verbose_json(raw: Dict[str, Any], path: Path) -> None:
    """Save raw transcription JSON."""
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_plain_text_from_verbose_json(raw: Dict[str, Any]) -> str:
    """Extract plain text from transcription response."""
    if "text" in raw and isinstance(raw["text"], str) and raw["text"].strip():
        return raw["text"].strip()
    # Fallback: concatenate segments
    segs = raw.get("segments") or []
    return "\n".join(seg.get("text", "").strip() for seg in segs if seg.get("text"))


def make_srt_from_verbose_json(raw: Dict[str, Any]) -> Optional[str]:
    """Generate SRT subtitle format from transcription segments."""
    segs = raw.get("segments")
    if not segs or not isinstance(segs, list):
        return None

    def sec_to_ts(sec: float) -> str:
        if sec < 0:
            sec = 0.0
        ms = int(round(sec * 1000))
        hh = ms // 3600000
        mm = (ms % 3600000) // 60000
        ss = (ms % 60000) // 1000
        mmm = ms % 1000
        return f"{hh:02}:{mm:02}:{ss:02},{mmm:03}"

    lines = []
    for i, seg in enumerate(segs, start=1):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", max(start + 0.5, start)))
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.append(str(i))
        lines.append(f"{sec_to_ts(start)} --> {sec_to_ts(end)}")
        lines.append(text)
        lines.append("")  # blank line

    return "\n".join(lines).strip() or None


def chat_clean_text(client: OpenAI, text: str, nlp_model: str, language_hint: str = "auto") -> str:
    """
    Use GPT to clean/punctuate/correct transcript while maintaining semantic fidelity.
    """
    sys_prompt = (
        "You are a professional transcription editor. Clean the provided transcript by: "
        "1) Fixing obvious typos and punctuation; 2) Proper sentence segmentation; "
        "3) Removing filler words and redundancy; 4) Preserving original meaning without embellishment; "
        "5) Maintaining foreign proper names; 6) Output in the same language as input."
    )
    user_prompt = (
        f"[Original Transcript]\n{text}\n\n"
        "Output the [Cleaned Text]. Only output the text itself without explanations."
    )

    def _call():
        return client.chat.completions.create(
            model=nlp_model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

    resp = backoff_retry(_call)
    return resp.choices[0].message.content.strip()


def chat_summarize(client: OpenAI, text: str, nlp_model: str, summary_lang: str = "auto") -> str:
    """
    Generate structured summary using GPT in specified language.
    """
    # Language-specific prompts
    if summary_lang == "zh":
        sys_prompt = (
            "你是一位专业的会议/讲座/视频摘要助手。"
            "请生成结构化的摘要，忠实地捕捉关键要点。"
            "无论输入文本是什么语言，都请用中文输出摘要。"
        )
        user_prompt = (
            "基于以下转录文本，请输出一个结构化的 Markdown 摘要，包括：\n"
            "1) 执行摘要（3-6个关键点）\n"
            "2) 详细要点（分条列出，必要时分小节）\n"
            "3) 重要引述/术语（如有）\n"
            "4) 行动项/下一步（如适用）\n"
            "5) 潜在风险或分歧点（如适用）\n\n"
            "[完整文本]\n"
            f"{text}\n\n"
            "请只输出中文 Markdown 内容。"
        )
    elif summary_lang == "en":
        sys_prompt = (
            "You are a professional meeting/lecture/video summarization assistant. "
            "Generate structured summaries that faithfully capture key points. "
            "Always output the summary in English regardless of the input language."
        )
        user_prompt = (
            "Based on the following transcript, output a structured Markdown summary including:\n"
            "1) Executive Summary (3-6 key points)\n"
            "2) Detailed Points (itemized, with subsections if needed)\n"
            "3) Important Quotes/Terms (if any)\n"
            "4) Action Items/Next Steps (if applicable)\n"
            "5) Potential Risks or Points of Disagreement (if applicable)\n\n"
            "[Full Text]\n"
            f"{text}\n\n"
            "Output only the English Markdown content."
        )
    else:  # auto - match input language
        sys_prompt = (
            "You are a professional meeting/lecture/video summarization assistant. "
            "Generate structured summaries that faithfully capture key points. "
            "Output the summary in the same language as the input text."
        )
        user_prompt = (
            "Based on the following transcript, output a structured Markdown summary including:\n"
            "1) Executive Summary (3-6 key points)\n"
            "2) Detailed Points (itemized, with subsections if needed)\n"
            "3) Important Quotes/Terms (if any)\n"
            "4) Action Items/Next Steps (if applicable)\n"
            "5) Potential Risks or Points of Disagreement (if applicable)\n\n"
            "[Full Text]\n"
            f"{text}\n\n"
            "Output the Markdown content in the same language as the input."
        )

    def _call():
        return client.chat.completions.create(
            model=nlp_model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

    resp = backoff_retry(_call)
    return resp.choices[0].message.content


# ------------------ Main Workflow ------------------

def main():
    parser = argparse.ArgumentParser(
        description="Video/Audio -> Transcription -> Cleaning -> Summary Pipeline"
    )
    parser.add_argument("input", type=str, help="Input video or audio file path")
    parser.add_argument("--out", type=str, default="output", help="Output directory")
    parser.add_argument("--asr-model", type=str, default="whisper-1",
                        help="ASR model (e.g., whisper-1, gpt-5-nano)")
    parser.add_argument("--nlp-model", type=str, default="gpt-5-nano",
                        help="GPT model for cleaning and summarization")
    parser.add_argument("--language", type=str, default="auto",
                        help="Language code for ASR (e.g., zh, en, or auto)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="WAV sample rate")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio extraction if audio.wav exists")
    parser.add_argument("--skip-clean", action="store_true", help="Skip text cleaning step, use raw transcript for summary")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing for faster transcription")
    parser.add_argument("--max-workers", type=int, default=3, help="Maximum number of parallel workers (default: 3)")
    parser.add_argument("--summary-lang", type=str, default="auto", choices=["auto", "zh", "en"],
                        help="Language for summary output: auto (match input), zh (Chinese), en (English)")
    args = parser.parse_args()

    load_dotenv()  # Load from .env file
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not found.", file=sys.stderr)
        print("Please set it in your environment or create a .env file.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Determine input type
    is_audio = is_audio_file(input_path)
    is_video = is_video_file(input_path)

    if not is_audio and not is_video:
        print(f"Error: File '{input_path.name}' is not a recognized video or audio format.", file=sys.stderr)
        print("Supported video formats: mp4, avi, mov, mkv, wmv, flv, webm, etc.", file=sys.stderr)
        print("Supported audio formats: mp3, wav, m4a, aac, ogg, flac, etc.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Output file paths
    audio_path = out_dir / "audio.wav"
    raw_json_path = out_dir / "transcript.raw.json"
    txt_path = out_dir / "transcript.txt"
    clean_path = out_dir / "transcript.clean.txt"
    srt_path = out_dir / "transcript.srt"
    summary_path = out_dir / "summary.md"

    # Step 1: Prepare audio
    if is_audio:
        # Input is already audio
        if input_path.suffix.lower() == '.wav' and args.skip_audio:
            print("Step 1/5: Using input audio file directly (WAV format)")
            audio_path = input_path
        else:
            print("Step 1/5: Converting audio to WAV format...")
            convert_audio_to_wav(input_path, audio_path, samplerate=args.sample_rate)
            print(f"  [OK] Audio converted to: {audio_path.relative_to(Path.cwd())}")
    else:
        # Input is video, extract audio
        if args.skip_audio and audio_path.exists():
            print("Step 1/5: Skipping audio extraction (--skip-audio flag and audio.wav exists)")
        else:
            print("Step 1/5: Extracting audio from video...")
            run_ffmpeg_extract_audio(input_path, audio_path, samplerate=args.sample_rate)
            print(f"  [OK] Audio extracted to: {audio_path.relative_to(Path.cwd())}")

    # Step 2: Transcribe audio
    print("Step 2/5: Transcribing audio (ASR)...")
    raw = transcribe_audio(client, audio_path, args.asr_model, language=args.language,
                          parallel=args.parallel, max_workers=args.max_workers)
    save_verbose_json(raw, raw_json_path)
    plain = extract_plain_text_from_verbose_json(raw)
    txt_path.write_text(plain, encoding="utf-8")
    print(f"  [OK] Transcript saved to: {txt_path.relative_to(Path.cwd())}")

    # Generate SRT if segments available
    srt = make_srt_from_verbose_json(raw)
    if srt:
        srt_path.write_text(srt, encoding="utf-8")
        print(f"  [OK] Subtitles saved to: {srt_path.relative_to(Path.cwd())}")
    else:
        print("  [INFO] No segments available, skipping SRT generation")

    # Step 3: Clean text (optional)
    if args.skip_clean:
        print("Step 3/5: Skipping text cleaning (--skip-clean flag)")
        cleaned = plain  # Use raw transcript for summary
    else:
        print("Step 3/5: Cleaning transcript (GPT)...")
        cleaned = chat_clean_text(client, plain, args.nlp_model, language_hint=args.language)
        clean_path.write_text(cleaned, encoding="utf-8")
        print(f"  [OK] Cleaned text saved to: {clean_path.relative_to(Path.cwd())}")

    # Step 4: Generate summary
    print("Step 4/5: Generating summary (GPT)...")
    summary_md = chat_summarize(client, cleaned, args.nlp_model, summary_lang=args.summary_lang)
    summary_path.write_text(summary_md, encoding="utf-8")
    print(f"  [OK] Summary saved to: {summary_path.relative_to(Path.cwd())}")

    # Step 5: Report results
    print("\n[SUCCESS] Processing complete! Output files:")
    print(f"  {out_dir.relative_to(Path.cwd())}/")
    print(f"     |-- audio.wav          - Processed audio")
    print(f"     |-- transcript.raw.json - Raw transcription data")
    print(f"     |-- transcript.txt      - Plain transcript")
    if not args.skip_clean:
        print(f"     |-- transcript.clean.txt - Cleaned transcript")
    if srt:
        print(f"     |-- transcript.srt      - Subtitle file")
    print(f"     |-- summary.md          - Structured summary")


if __name__ == "__main__":
    main()