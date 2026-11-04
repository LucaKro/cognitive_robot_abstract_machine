# cramspeech

Speech recognition for CRAM. When asked to, it records from a microphone for a given time and
transcribes the recording with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

```python
from cramspeech.microphone import Microphone
from cramspeech.speech_recognizer import SpeechRecognizer
from cramspeech.transcription import WhisperTranscriber

recognizer = SpeechRecognizer(microphone=Microphone(), transcriber=WhisperTranscriber())
print(recognizer.hear(duration=5.0).text)
```

## Setup

- The microphone is read through [sounddevice](https://python-sounddevice.readthedocs.io), which
  needs the PortAudio library on Linux: `sudo apt install libportaudio2`.
- `python -m sounddevice` lists the input devices. Pass an index or name as `Microphone(device=...)`.
- The Whisper model is downloaded from Hugging Face on the first transcription, not at install
  time, and cached in `~/.cache/huggingface/hub`. To fetch it beforehand:
  `python -c "import faster_whisper; faster_whisper.download_model('small')"`.
