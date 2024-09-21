import yt_dlp
import whisper
import torch
import os
import re

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
        'postprocessors': [{  # Ensure we get an audio file in a common format
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
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

    audio_file = 'audio.mp3'  # Since we set preferredcodec to mp3

    if not os.path.exists(audio_file):
        print("Error: Audio file not found.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading Whisper model...")
    model = whisper.load_model("small", device=device)  # Use a better model for accuracy

    print("Transcribing audio...")
    result = model.transcribe(audio_file, fp16=False if device == "cpu" else True)
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
