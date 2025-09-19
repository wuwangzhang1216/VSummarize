# VSummarize - Video/Audio to Summary Pipeline

A powerful, production-ready tool for transcribing and summarizing video/audio content using OpenAI's API. Features parallel processing, multi-language support, and automatic chunking for large files.

## 🚀 Key Features

- **Universal Media Support**: Process video (MP4, AVI, MOV, etc.) and audio (MP3, WAV, M4A, etc.) files
- **Smart Chunking**: Automatically splits large files (>25MB) for API compliance
- **Parallel Processing**: 2-3x faster transcription with concurrent chunk processing
- **Cross-Language Summaries**: Generate summaries in Chinese or English regardless of input language
- **Flexible Pipeline**: Optional text cleaning step for speed vs quality tradeoff
- **SRT Subtitles**: Auto-generated subtitle files with precise timestamps
- **Progress Tracking**: Real-time progress bars for long-running operations
- **Robust Error Handling**: Automatic retries with exponential backoff

## 📋 Prerequisites

1. **FFmpeg** (Required for media processing)
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg`

2. **Python 3.8+**
   - Download from [python.org](https://python.org) if not installed

3. **OpenAI API Key**
   - Get your key from [OpenAI Platform](https://platform.openai.com/api-keys)
   - Requires active OpenAI account with API credits

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/VSummarize.git
cd VSummarize

# Install dependencies
pip install -r requirements.txt

# Set up API key
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## 📖 Usage

### Basic Usage

```bash
# Process a video file
python vid2summary.py video.mp4

# Process an audio file
python vid2summary.py podcast.mp3

# Specify output directory
python vid2summary.py input.mp4 --out results/
```

### Advanced Options

```bash
python vid2summary.py input.mp4 \
  --out output_folder \
  --asr-model whisper-1 \
  --nlp-model gpt-4o \
  --language zh \
  --summary-lang en \
  --parallel \
  --max-workers 3 \
  --skip-clean
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | Required | Path to video or audio file |
| `--out` | `output` | Output directory for results |
| `--asr-model` | `whisper-1` | ASR model (whisper-1, gpt-4o-audio-preview) |
| `--nlp-model` | `gpt-4o` | GPT model for text processing |
| `--language` | `auto` | Transcription language (auto, en, zh, etc.) |
| `--summary-lang` | `auto` | Summary output language (auto, zh, en) |
| `--sample-rate` | `16000` | Audio sample rate in Hz |
| `--skip-audio` | False | Skip audio extraction if exists |
| `--skip-clean` | False | Skip text cleaning for faster processing |
| `--parallel` | False | Enable parallel chunk processing |
| `--max-workers` | `3` | Number of parallel workers (2-4 recommended) |

## 📁 Output Files

```
output/
├── audio.wav              # Processed audio (WAV format)
├── transcript.raw.json    # Raw transcription with metadata
├── transcript.txt         # Plain text transcript
├── transcript.clean.txt   # Cleaned transcript (if not skipped)
├── transcript.srt         # Subtitle file with timestamps
└── summary.md            # Structured Markdown summary
```

### Summary Structure

Generated summaries include:
1. **Executive Summary** - 3-6 key points
2. **Detailed Points** - Itemized details with subsections
3. **Important Quotes/Terms** - Key terminology and direct quotes
4. **Action Items** - Next steps and TODOs
5. **Risks/Issues** - Potential problems or disagreements

## 💡 Examples

### Process a meeting recording
```bash
python vid2summary.py meeting.mp4 --out meetings/2024-01
```

### Chinese lecture with English summary
```bash
python vid2summary.py lecture.mp4 --language zh --summary-lang en
```

### Fast processing (parallel + skip cleaning)
```bash
python vid2summary.py podcast.mp3 --parallel --skip-clean
```

### Process existing audio file
```bash
python vid2summary.py audio.wav --skip-audio
```

## ⚡ Performance Optimization

### Parallel Processing
For files >25MB that require chunking:
- **Sequential**: ~30 seconds per chunk
- **2 workers**: ~50% faster
- **3 workers**: ~65% faster
- **4 workers**: ~70% faster (may hit rate limits)

### Speed Tips
1. Use `--parallel` for large files
2. Add `--skip-clean` if transcript quality is good
3. Use `--skip-audio` when reprocessing
4. Set `--max-workers 2-3` for optimal balance

## 🌍 Language Support

### Transcription Languages
Supports 50+ languages including:
- English (en)
- Chinese (zh)
- Spanish (es)
- French (fr)
- German (de)
- Japanese (ja)
- Korean (ko)
- And many more...

### Summary Languages
- **auto**: Matches input language
- **zh**: Forces Chinese output
- **en**: Forces English output

## 💰 API Costs

Estimated costs (as of 2024):
- **Whisper API**: ~$0.006 per minute of audio
- **GPT-4o**: ~$0.01-0.03 per summary
- **Total**: ~$0.02-0.05 per minute of content

## 🔍 Troubleshooting

### Common Issues

**FFmpeg not found**
- Ensure FFmpeg is installed: `ffmpeg -version`
- Add FFmpeg to system PATH

**API key errors**
- Check `.env` file has correct key
- Verify API credits available

**File too large**
- Tool auto-chunks files >25MB
- Use `--parallel` for faster processing

**Encoding issues on Windows**
- Output uses UTF-8 encoding
- May show encoding errors in some terminals

## 📝 Supported Formats

### Video Formats
MP4, AVI, MOV, MKV, WMV, FLV, WebM, M4V, MPG, MPEG, 3GP, TS

### Audio Formats
MP3, WAV, M4A, AAC, OGG, FLAC, WMA, OPUS, AIFF, APE

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- OpenAI for Whisper and GPT APIs
- FFmpeg team for media processing
- Community contributors

## 📧 Support

For issues or questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Include error messages and system info

---

**Note**: This tool requires an active OpenAI API subscription. Costs vary based on usage.