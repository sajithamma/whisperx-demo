import whisperx
import torch
import gc
from pyannote.audio import Pipeline
from pyannote.core import Segment
from pyannote.audio import Audio
import speechbrain

# Verify that speechbrain is installed
print(f"SpeechBrain version: {speechbrain.__version__}")

# Set the device to CPU for compatibility
device = 'cpu'

audio_file = "/Path/to/file.wav"
batch_size = 16  # reduce if low on GPU mem
compute_type = "float32"  # change to "int8" if low on GPU mem (may reduce accuracy)
language = "en"
hf_token = "<YourHuggingFaceToken>"  # Replace this with your actual Hugging Face token

# 1. Load the model with translation task
model = whisperx.load_model("large-v2", device=device, compute_type=compute_type, task='translate', language=language)

# 2. Load the audio
audio = whisperx.load_audio(audio_file)

# 3. Perform transcription
result = model.transcribe(audio, batch_size=batch_size)
print("Transcription completed.")
print(result["segments"])  # before alignment

# 4. Align Whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
print("Alignment completed.")
print(result["segments"])  # after alignment

# 5. Load PyAnnote speaker diarization pipeline
try:
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
    print("Pipeline loaded successfully.")
except ImportError as e:
    print(f"Error loading pipeline: {e}")

# 6. Perform speaker diarization on the entire audio file
diarization_result = pipeline(audio_file)
print("Diarization completed.")

# Print the diarization results
for speech_turn, _, speaker in diarization_result.itertracks(yield_label=True):
    print(f"Speaker {speaker}: from {speech_turn.start:.1f}s to {speech_turn.end:.1f}s")

# Optional: align the transcribed segments with the diarization results
for segment in result["segments"]:
    start_time = segment["start"]
    end_time = segment["end"]
    for speech_turn, _, speaker in diarization_result.itertracks(yield_label=True):
        if start_time >= speech_turn.start and end_time <= speech_turn.end:
            segment["speaker"] = speaker
            break

# Print the translated segments with speaker info
for segment in result["segments"]:
    speaker = segment.get("speaker", "Unknown")
    print(f"Speaker {speaker}: {segment['text']}")

# Clean up
gc.collect()
torch.cuda.empty_cache()
