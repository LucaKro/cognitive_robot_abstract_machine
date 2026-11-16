# cramspeech

Speech for CRAM. It listens to a microphone, finds where somebody spoke and paused
with the Silero voice activity model, and transcribes each utterance with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). The command `nlp-listen` publishes
every heard sentence on ROS, and `PlanModel` turns a heard instruction into plans.

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

## Plans from an instruction

`PlanModel` turns a spoken instruction into plans with a trained language model in GGUF format,
run by [llama.cpp](https://github.com/ggml-org/llama.cpp):

```python
from pathlib import Path

from cramspeech.inference.plan_model import PlanModel

plan_model = PlanModel(model_path=Path("plan-q4.gguf"))

transcript = recognizer.hear_next_utterance(timeout=10.0)
instruction_plans = plan_model.write_plans(transcript.text)
for plan in instruction_plans.plans:            # one plan per task of the instruction
    print(plan.summary, plan.steps)
```

- The model can only write what [`plan_grammar.gbnf`](src/cramspeech/inference/resources/plan_grammar.gbnf)
  allows: the plan format it was trained on, with any words for objects, locations and people.
- The model is loaded on the first instruction and then kept.

## Setup

- `cramspeech` is not part of a plain `uv sync`. Install it from the repository root with
  `uv sync --extra speech --active --inexact`, where `--inexact` keeps what is already installed.
- `PlanModel` runs on llama-cpp-python, which the CRAM workspace already installs for robokudo.
- The microphone is read through [sounddevice](https://python-sounddevice.readthedocs.io), which
  needs the PortAudio library on Linux: `sudo apt install libportaudio2`.
- `nlp-listen --list-devices` (or `InputDevice.list_available()`) lists the input devices. Pass an
  index or a unique part of a name as `Microphone(device=...)` or `nlp-listen --device ...`.
- The Whisper model is downloaded from Hugging Face on the first transcription, not at install
  time, and cached in `~/.cache/huggingface/hub` (`small` is about 480 MB). To fetch it
  beforehand: `python -c "import faster_whisper; faster_whisper.download_model('small')"`.
- `nlp-listen` needs a sourced ROS 2 installation. The Python API works without ROS.
- As installed, llama-cpp-python runs `PlanModel` on the CPU. On a laptop with an NVIDIA graphics
  card and the CUDA toolkit, rebuild llama-cpp-python once with CUDA switched on, and the model
  runs on the graphics card from then on:

  ```bash
  CMAKE_ARGS="-DGGML_CUDA=on" uv sync --extra speech --active --inexact \
      --reinstall-package llama-cpp-python --no-cache
  ```

  Run it again whenever llama-cpp-python is updated, since a new version is built for the CPU
  again. `PlanModel` needs no setting for either: `ComputeDevice.GPU`, the default, uses the
  graphics card when the build has CUDA and the CPU otherwise.
