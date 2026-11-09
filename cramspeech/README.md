# cramspeech

Speech recognition for CRAM. It listens to a microphone, finds where somebody spoke and paused
with the Silero voice activity model, and transcribes each utterance with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). The command `nlp-listen` publishes
every heard sentence on ROS.

## Listening on ROS

```bash
source /opt/ros/jazzy/setup.bash   # rclpy and std_msgs come from the ROS installation
nlp-listen --list-devices          # the input devices, the default one starred
nlp-listen --device <name>         # any unique part of a listed name
nlp-listen --echo                  # the robot also says every heard sentence back
```

`nlp-listen` listens from its start until it is stopped with Ctrl+C or SIGTERM. Each utterance
it heard words in goes out as a `std_msgs/String`:

| Topic | For | Published |
|---|---|---|
| `whisper_out` | anyone who acts on what was said | always |
| `/tts_text` | the robot's `tts_node`, which says the sentence back | only with `--echo` |

Options: `--model-size` (`tiny`, `base`, `small`, default `small`), `--compute-type`
(`int8`, `float16`, `float32`, default `int8`) and `--language` (`en`, `de`, default `en`).
`nlp-listen --help` lists them.

## From Python

```python
from cramspeech.microphone import Microphone
from cramspeech.speech_recognizer import SpeechRecognizer
from cramspeech.transcription import WhisperTranscriber

recognizer = SpeechRecognizer(microphone=Microphone(device=6), transcriber=WhisperTranscriber())

print(recognizer.hear(duration=5.0).text)                # records for 5 s
print(recognizer.hear_next_utterance(timeout=10.0).text) # waits for one utterance
for transcript in recognizer.hear_utterances():          # every utterance, until you stop
    print(transcript.text)
```

`hear_next_utterance` raises `NothingHeard` when nobody starts speaking within the timeout. All
listening raises `MicrophoneStopped` when the device stops delivering audio.

## Setup

- `cramspeech` is not part of a plain `uv sync`. Install it from the repository root with
  `uv sync --extra speech --active --inexact`, where `--inexact` keeps what is already installed.
- The microphone is read through [sounddevice](https://python-sounddevice.readthedocs.io), which
  needs the PortAudio library on Linux: `sudo apt install libportaudio2`.
- `nlp-listen --list-devices` (or `InputDevice.list_available()`) lists the input devices. Pass an
  index or a unique part of a name as `Microphone(device=...)` or `nlp-listen --device ...`.
- The Whisper model is downloaded from Hugging Face on the first transcription, not at install
  time, and cached in `~/.cache/huggingface/hub` (`small` is about 480 MB). To fetch it
  beforehand: `python -c "import faster_whisper; faster_whisper.download_model('small')"`.
- `nlp-listen` needs a sourced ROS 2 installation. The Python API works without ROS.
