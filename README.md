# Whisperx Demo

You can install the Whisperx package from the [Python Package Index](https://pypi.org/project/whisperx/).

## Setup Conda Environment
```bash
conda create --name whisperx python=3.10
conda activate whisperx
```

```bash
pip install whisperx
```

or you can install the latest version from the [GitHub repository]

```bash
git clone git@github.com:m-bain/whisperX.git
cd whisperX
pip install -e .
```

## Test 
```bash
whisperx examples/sample01.wav
```

## Python Code Example
```python

device = 'cpu' #or cuda if available
audio_file = "/Path/to/'file.wav"
batch_size = 16  # reduce if low on GPU mem
compute_type = "float32"  # change to "int8" if low on GPU mem (may reduce accuracy)
language = "en"

model = whisperx.load_model("large-v2", device=str(device), compute_type=compute_type, task='translate', language=language)

# 2. Load the audio
audio = whisperx.load_audio(audio_file)

# 3. Perform translation
result = model.transcribe(audio, batch_size=batch_size)
```

## Models Available
* tiny
* base
* small
* medium
* large

Note: 

* Live translation is kinda super fast, with base model on CPU, so we can use whisperx.
* For live transcription, with large model (more accurate detection, we need GPU, tiny and base model, CPU is enough, nearly 90% accuracy, for words, some words are tricky, with large model, all words are detecting good , but GPU is recommended)
* Speaker detection is process heavy, CPU is not enough, need GPU machine for the best performance
* Translation happens in less than second (Mac M1, CPU), speaker detection takes around 30 seconds ( CPU, Mac M1)