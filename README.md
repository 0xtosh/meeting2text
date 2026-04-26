# Meeting2Text

Meeting2Text is a Python-based utility to convert video e.g. meeting recordings into clean text transcripts and structured summaries. It supports two distinct operational modes: a local mode using Faster-Whisper for private, on-device processing, and a cloud mode utilizing NVIDIA NIM free models for transcription and summarization.

## Features

- Local transcription using Faster-Whisper (CPU-based).
- Cloud transcription and summarization via NVIDIA NIM API.
- Automatic audio extraction and splitting using FFmpeg.
- Fallback model support for cloud resiliency.
- Pure text output for easy integration with other tools.
- Detailed phase-by-phase timing and performance metrics.

## Dependencies

### System Requirements
This program requires FFmpeg to be installed and available in your system PATH for audio extraction and splitting.

### Python Libraries
Install the required Python dependencies via pip:

pip install requests faster-whisper tqdm

Note: Local mode requires approximately 1.5 GB of disk space to download the default "medium.en" Whisper model upon first use.

## Installation

1. Clone or download the meeting2text.py script to your local machine.
2. Ensure FFmpeg is installed. You can verify this by running "ffmpeg -version" in your terminal.
3. Install the Python dependencies listed above.
4. (Optional) For cloud mode, obtain an API key from the NVIDIA NIM platform.

## Usage Examples

### 1. Local Mode
Local mode processes the audio entirely on your CPU. This is the most private option as no data leaves your machine.

python meeting2text.py mario.mp4 --mode local

### 2. Cloud Mode
Cloud mode uploads compressed audio chunks to NVIDIA's servers to leverage high-performance multimodal models. This mode also generates a structured summary.

python meeting2text.py mario.mp4 --mode cloud --api-key YOUR_NVIDIA_API_KEY

## Performance Benchmarks

Performance between cloud and CPU modes can differ a lot depending on your local hardware and network conditions. In the benchmark data provided below for a 6-minute video file, the total elapsed time is almost the same. This is common when using free-tier cloud models that can be lower in performance or experience higher latency during the reasoning-heavy summary phase compared to local CPU transcription.

### Benchmark Data: 6-Minute Video File

#### Cloud Mode Performance (Total: 5m 59s)
Cloud mode transcribes audio chunks very quickly but the summary generation for a full meeting can take several minutes on the free tier.
``` 
Phase timings:
  Audio extraction             2s
  Audio splitting              1s
  Transcription             1m 24s  (2 chunks)
  Summary                   4m 31s
--------------------------------------------------------------
  Total elapsed             5m 59s
``` 
#### Local Mode Performance (Total: 7m 30s)
Local mode speed is determined entirely by your CPU. It transcribes the file in one continuous pass and does not generate a summary. Benchmark CPU: 12-Core / 24-Thread High-Performance Desktop CPU @ 3.8 GHz
``` 
Phase timings:
  Audio extraction             0s
  Model load                   7s
  Transcription             7m 17s
--------------------------------------------------------------
  Total elapsed             7m 30s
``` 
## Data Privacy

When using Cloud Mode, your audio data is uploaded to NVIDIA's servers and processed by third-party models (Microsoft Phi-4 and Mistral Large). Do not proceed if your recording contains confidential, sensitive, or personally identifiable information that must not leave your organisation's systems. Local Mode is recommended for all sensitive data.

## Output

The program generates two primary files in the same directory as the input:
1. [filename].txt: The full verbatim transcript.
2. [filename]_summary.txt: A structured meeting summary (Cloud Mode only).

## Benchmark Example



### Local mode using the CPU and "Whisper" model

Benchmark CPU: 12-Core / 24-Thread High-Performance Desktop CPU @ 3.8 GHz
``` 
C:\dojo\meeting2text>python meeting2text.py --mode local mario.mp4
==============================================================
  Meeting Transcriber  (local / faster-whisper edition)
==============================================================

[1/4] Extracting audio -> mario.wav
      cmd: ffmpeg -y -i C:\dojo\meeting2text\mario.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 C:\dojo\meeting2text\mario.wav
      WAV size : 10.9 MB
      phase      0s  |  total      0s

[2/4] Loading Whisper model 'medium.en' (int8, CPU)...
      phase      7s  |  total     12s

[3/4] Analysing and transcribing...
  [2.7s] So, you have a movie for me?
  [4.7s] Yes, sir, I do.
  [5.7s] I was thinking it's about time we make an animated Super Mario movie.
  [9.7s] Oh my God, yeah, that is a massive franchise.
  [11.7s] How have we not attempted that yet?
  [13.7s] Cause the live action one from 30 years ago made Nintendo lose trust in the entire film industry.
  [18.7s] Yeah, that's fair.

  <cut for formatting reasons> 
 
 99%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉ | 356s/358s
      phase  7m 17s  |  total  7m 30s

DONE! Total elapsed 7m 30s
``` 

## Cloud mode using free Nvidia models
``` 
C:\dojo\meeting2text>python meeting2text.py --mode cloud mario.mp4 --api-key <api key>

  CLOUD MODE -- DATA PRIVACY NOTICE
  ──────────────────────────────────────────────────────────
  Your audio will be uploaded to NVIDIA's servers and
  processed by the following third-party models:
    Transcription : microsoft/phi-4-multimodal-instruct
    Summary       : mistralai/mistral-large-3-675b-instruct-2512
  Endpoint : https://integrate.api.nvidia.com/v1
  File     : mario.mp4

  Do not proceed if your recording contains confidential,
  sensitive, or personally identifiable information that
  must not leave your organisation's systems.
  ──────────────────────────────────────────────────────────
  Type 'yes' to confirm and continue, anything else to abort: yes

==============================================================
  Meeting Transcriber  (NVIDIA NIM edition)
==============================================================
  Input          : C:\dojo\meeting2text\mario.mp4
  Transcript out : C:\dojo\meeting2text\mario.txt
  Summary out    : C:\dojo\meeting2text\mario_summary.txt
  Chunk size     : 5 min  (300 s)
  Transcribe     : microsoft/phi-4-multimodal-instruct
  Summarise      : mistralai/mistral-large-3-675b-instruct-2512
  ffmpeg         : ffmpeg
  Started        : 2026-04-26 22:29:22
==============================================================

[1/4] Extracting audio -> mario.wav
      cmd: ffmpeg -y -i C:\dojo\meeting2text\mario.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 C:\dojo\meeting2text\mario.wav
      WAV size : 10.9 MB
      phase      0s  |  total      2s

[2/4] Splitting audio into 5-minute chunks...
      2 chunks created
      phase      1s  |  total      3s

[3/4] Transcribing 2 chunks via microsoft/phi-4-multimodal-instruct...
      (Rate limit: 40 req/min - sleeping 1.5 s between calls)

      [1/2] chunk_0000.wav (1172 KB, 18372 chars)  phase  1m 19s  |  total  1m 22s
      [2/2] chunk_0001.wav (227 KB, 936 chars)  phase      3s  |  total  1m 27s
      phase  1m 24s  |  total  1m 27s

      Transcript saved -> C:\dojo\meeting2text\mario.txt
      Total length     : 19,310 chars  /  ~3,467 words

[4/4] Generating structured summary with mistralai/mistral-large-3-675b-instruct-2512...
      Transcript length : 19,310 chars  / ~3,467 words
      phase  4m 31s  |  total  5m 59s
      Summary saved    -> C:\dojo\meeting2text\mario_summary.txt
      WAV deleted      -> mario.wav

══════════════════════════════════════════════════════════════
  Done!  2026-04-26 22:35:20
──────────────────────────────────────────────────────────────
  Phase timings:
    Audio extraction            2s
    Audio splitting             1s
    Transcription            1m 24s  (2 chunks)
    Summary                  4m 31s
──────────────────────────────────────────────────────────────
  Total elapsed            5m 59s
──────────────────────────────────────────────────────────────
  Outputs:
    C:\dojo\meeting2text\mario.txt
    C:\dojo\meeting2text\mario_summary.txt
══════════════════════════════════════════════════════════════
``` 



