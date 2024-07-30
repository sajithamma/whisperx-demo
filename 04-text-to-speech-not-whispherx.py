import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from TTS.api import TTS

# Enable MPS fallback to CPU for unsupported operations
# Do this for mac
# pip install numpy
# pip install TTS
# export CFLAGS="-I$(python -c 'import numpy; print(numpy.get_include())')"
# export LDFLAGS="-L/usr/local/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/lib"


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.to(device)

# Run TTS
while True:
    text = input("Enter the text to convert to speech: ")
    if text == "exit":
        break
    tts.tts_to_file(text=text, speaker_wav="/Path/to/your/reference/sound/mysound.wav", language="en", file_path="output.wav")
    print("Speech file generated successfully")
