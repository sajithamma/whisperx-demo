import whisperx
import gc
import torch

# Set the device to use MPS if available, otherwise CPU
device = 'cpu' #or cuda if available

audio_file = "/Path/to/'file.wav"
batch_size = 16  # reduce if low on GPU mem
compute_type = "float32"  # change to "int8" if low on GPU mem (may reduce accuracy)
language = "en"

# 1. Load the model with translation task
model = whisperx.load_model("large-v2", device=str(device), compute_type=compute_type, task='translate', language=language)

# Optional: save model to local path
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device=str(device), compute_type=compute_type, download_root=model_dir, task='translate')

# 2. Load the audio
audio = whisperx.load_audio(audio_file)

# 3. Perform translation
result = model.transcribe(audio, batch_size=batch_size)

# 4. Print the translated segments
print(result["segments"])  # before alignment

