import yt_dlp
import whisper
import torch
import os
import re
import warnings

def sanitize_filename(filename):
    # Remove invalid characters for filenames
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def main():
    url = input("Enter YouTube video URL: ")

    ydl_opts = {
        'format': 'worstaudio/worst',  # Download the lowest quality audio only
        'outtmpl': 'audio.%(ext)s',
        'noplaylist': True,
        'quiet': True,
        'postprocessors': [{  # Ensure we get an audio file in a fast-to-process format
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',  # Use WAV for faster processing
            'preferredquality': '64',  # Lower quality for faster download
        }],
    }

    print("Downloading audio and retrieving video info...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Extract video information and download audio
        info_dict = ydl.extract_info(url, download=True)
        video_title = info_dict.get('title', 'Untitled')

    # Sanitize the video title to create a valid filename
    sanitized_title = sanitize_filename(video_title)

    audio_file = 'audio.wav'  # Since we set preferredcodec to wav

    if not os.path.exists(audio_file):
        print("Error: Audio file not found.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading Whisper model...")
    # Suppress the FutureWarning
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Use a smaller model for faster transcription
    model = whisper.load_model("base", device=device)

    print("Transcribing audio...")
    # Set parameters to speed up transcription
    result = model.transcribe(
        audio_file,
        fp16=False if device == "cpu" else True,
        condition_on_previous_text=False,  # Speeds up decoding
        beam_size=1,  # Simplifies the search
        best_of=1,    # Reduces the number of candidates
        verbose=True 
    )
    transcription = result['text']

    # Create the transcription filename using the sanitized video title
    transcription_filename = f"Transcribed - {sanitized_title}.txt"

    with open(transcription_filename, "w", encoding='utf-8') as f:
        f.write(transcription)

    print(f"Transcription saved to {transcription_filename}")

    # Delete the temporary audio file
    os.remove(audio_file)
    print("Temporary audio file deleted.")

if __name__ == "__main__":
    main()
