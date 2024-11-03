# app.py
import streamlit as st
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import whisper
import os

def save_uploaded_file(uploaded_file):
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def extract_audio(video_path):
    try:
        video_clip = VideoFileClip(video_path)
        audio_path = os.path.splitext(video_path)[0] + ".wav"
        video_clip.audio.write_audiofile(audio_path)
        video_clip.close()
        return audio_path
    except Exception as e:
        st.error(f"Error extracting audio: {e}")
        return None

def split_audio(audio_path, chunk_length_ms=600000):
    try:
        audio = AudioSegment.from_file(audio_path)
        chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        chunk_files = []
        for idx, chunk in enumerate(chunks):
            chunk_filename = f"audio_chunk_{idx}.wav"
            chunk.export(chunk_filename, format="wav")
            chunk_files.append(chunk_filename)
        return chunk_files
    except Exception as e:
        st.error(f"Error splitting audio: {e}")
        return []

def transcribe_audio(chunks):
    try:
        model = whisper.load_model("base")  # Change to 'small', 'medium', etc., as needed
        transcripts = []
        for chunk in chunks:
            result = model.transcribe(chunk)
            transcripts.append(result["text"])
        full_transcript = " ".join(transcripts)
        return full_transcript
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return ""

def compress_video(video_path, compressed_path, target_size_mb=500):
    try:
        video_clip = VideoFileClip(video_path)
        # Calculate the target bitrate
        duration = video_clip.duration  # in seconds
        target_bitrate = (target_size_mb * 8 * 1000) / duration  # in kbps
        video_clip.write_videofile(
            compressed_path,
            bitrate=f"{int(target_bitrate)}k",
            codec="libx264",
            audio_codec="aac"
        )
        video_clip.close()
        return compressed_path
    except Exception as e:
        st.error(f"Error compressing video: {e}")
        return None

# Streamlit app
st.title("StreamScribe - Video Processing and Transcription")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.write("Uploading and processing video...")
    video_path = save_uploaded_file(uploaded_file)
    
    # Optional: Compress the video to manage large file sizes
    compressed_path = os.path.splitext(video_path)[0] + "_compressed.mp4"
    compressed_video = compress_video(video_path, compressed_path)
    if compressed_video:
        st.success("Video compressed successfully.")
        video_path_to_use = compressed_video
    else:
        st.warning("Using original video due to compression issues.")
        video_path_to_use = video_path
    
    # Extract audio
    audio_path = extract_audio(video_path_to_use)
    if audio_path:
        st.success("Audio extracted successfully.")
        
        # Split audio into chunks
        chunks = split_audio(audio_path)
        if chunks:
            st.success("Audio split into chunks successfully.")
            
            # Transcribe audio
            with st.spinner("Transcribing audio..."):
                transcript = transcribe_audio(chunks)
            if transcript:
                st.success("Transcription completed.")
                st.text_area("Transcript", transcript, height=300)
                
                # Optionally, save the transcript to a file or pass it to the next stage
            else:
                st.error("Transcription failed.")
        else:
            st.error("Failed to split audio into chunks.")
    else:
        st.error("Audio extraction failed.")
