import whisperx
import sounddevice as sd
import numpy as np
import queue
import torch
import threading
import warnings
import time
from whisperx.vad import load_vad_model

# Set the device to CPU for compatibility
device = 'cpu' #or cuda if available

# Parameters
sample_rate = 16000
chunk_duration = 1.5  # seconds
batch_size = 16  # reduce if low on GPU mem
compute_type = "float32"  # change to "int8" if low on GPU mem (may reduce accuracy)
model_name = "base"
task = 'translate'
language = "en"
vad_onset = 0.500
vad_offset = 0.363
silence_threshold = 1.5  # seconds of silence to stop

warnings.filterwarnings("ignore", category=UserWarning, message="audio is shorter than 30s")
warnings.filterwarnings("ignore", category=UserWarning, message="Model was trained with pyannote.audio")
warnings.filterwarnings("ignore", category=UserWarning, message="Model was trained with torch")

# Initialize a queue to hold the audio chunks
audio_queue = queue.Queue()

# Load the model with the translation task
model = whisperx.load_model(model_name, device=device, compute_type=compute_type, task=task, language=language)

vad_pipeline = load_vad_model(device, vad_onset=vad_onset, vad_offset=vad_offset)

# Variable to track the last time speech was detected
last_speech_time = time.time()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    # Flatten the array and put it in the queue
    audio_queue.put(indata.copy().flatten())

def process_audio():
    global last_speech_time
    print("Processing audio...")
    while True:
        if not audio_queue.empty():
            audio_chunk = audio_queue.get()
            audio_chunk = np.float32(audio_chunk)
            
            # Transcribe the audio chunk
            result = model.transcribe(audio_chunk, batch_size=batch_size)
            
            # Check if any speech was detected
            if result["segments"]:
                last_speech_time = time.time()
                # Print the translated segments
                for segment in result["segments"]:
                    print(segment["text"])
            else:
                print("No speech detected in this chunk")

            # Check for silence duration
            if time.time() - last_speech_time > silence_threshold:
                print(f"Detected silence for {silence_threshold} seconds, stopping...")
                return  # This will end the thread

# Start a new thread to process the audio chunks
processing_thread = threading.Thread(target=process_audio)
processing_thread.daemon = True
processing_thread.start()

# Start the audio stream
with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=int(sample_rate * chunk_duration)):
    print("Listening... Press Ctrl+C to stop.")
    try:
        while processing_thread.is_alive():
            sd.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")

# Clean up
del model
torch.cuda.empty_cache()