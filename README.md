# DS_StreamScribe

Unlock insights, chat with your videos, and navigate with ease.

### **Project Setup Using Poetry**

#### **Prerequisites**

- Python 3.7 or higher
- Poetry (for dependency management)

#### **Installation Steps**

1. **Clone the Repository**

   ```bash
   git clone <REMOTE_URL>
   cd streamscribe
   ```

2. **Install Poetry**

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install Dependencies**

   ```bash
   poetry install
   ```

4. **Activate the Virtual Environment**

   ```bash
   poetry shell
   ```

5. **Run the Application**

   ```bash
   poetry run streamlit run app.py
   ```

---




# **Project Name: StreamScribe**

*"Unlock insights, chat with your videos, and navigate with ease."*

---

## **Project Overview**

**StreamScribe** is a Streamlit application that allows users to:

1. **Upload Long Videos**: Handle videos up to an hour and a half, with efficient processing techniques.
2. **Transcribe Audio**: Use OpenAI's Whisper model to convert speech to text.
3. **Summarize the Video**: Generate concise summaries of the video's content.
4. **Chat with the Video Content**: Enable an interactive chat interface where users can ask questions about the video.
5. **Generate Headlines with Timestamps**: Extract key points with timings and allow users to jump to specific parts of the video.

---

## **Team Roles and Instructions**

### **Team Member 1: Backend Developer (Video Processing and Transcription)**

**Responsibilities**:

- Handle video upload and preprocessing.
- Extract audio from videos.
- Transcribe audio using Whisper.
- Optimize processing for long videos.

**Tasks**:

1. **Set Up the Development Environment**:

   - Install necessary libraries:
     ```bash
     pip install streamlit moviepy ffmpeg-python pydub whisper torch
     ```
   - Ensure FFmpeg is installed on your system.

2. **Video Upload and Audio Extraction**:

   - **Implement Video Upload**:
     - Use Streamlit's `st.file_uploader()` to allow users to upload video files.
     - Handle large file uploads by increasing Streamlit's file size limit if necessary.
   - **Extract Audio from Video**:
     - Use `moviepy` or `ffmpeg-python` to extract audio.
     - Example:
       ```python
       from moviepy.editor import VideoFileClip

       def extract_audio(video_file):
           video_clip = VideoFileClip(video_file)
           audio_clip = video_clip.audio
           audio_path = "extracted_audio.wav"
           audio_clip.write_audiofile(audio_path)
           return audio_path
       ```

3. **Transcribe Audio with Whisper**:

   - **Process Long Audio Files**:
     - Split audio into smaller chunks (e.g., 10-minute segments) to manage memory usage.
     - Example:
       ```python
       from pydub import AudioSegment

       def split_audio(audio_path, chunk_length_ms=600000):
           audio = AudioSegment.from_file(audio_path)
           chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
           chunk_files = []
           for idx, chunk in enumerate(chunks):
               chunk_filename = f"audio_chunk_{idx}.wav"
               chunk.export(chunk_filename, format="wav")
               chunk_files.append(chunk_filename)
           return chunk_files
       ```
   - **Transcribe Chunks**:
     - Use Whisper to transcribe each chunk.
     - Example:
       ```python
       import whisper

       def transcribe_audio(chunks):
           model = whisper.load_model("base")  # Choose model size based on performance needs
           transcripts = []
           for chunk in chunks:
               result = model.transcribe(chunk)
               transcripts.append(result["text"])
           full_transcript = " ".join(transcripts)
           return full_transcript
       ```

4. **Optimization**:

   - **Compression and Efficient Storage**:
     - Compress audio files if necessary.
     - Use efficient data structures to handle transcripts and timestamps.
   - **Error Handling**:
     - Implement try-except blocks to handle potential errors during processing.

---

### **Team Member 2: NLP Engineer (Summarization and Headline Generation)**

**Responsibilities**:

- Summarize the transcribed text.
- Generate headlines with timestamps.
- Ensure accurate alignment between text and video timings.

**Tasks**:

1. **Text Summarization**:

   - **Use Pre-trained Models**:
     - Utilize models like T5 or BART from Hugging Face Transformers.
     - Example:
       ```python
       from transformers import pipeline

       summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

       def summarize_text(text):
           summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
           return summary[0]['summary_text']
       ```
   - **Handle Long Texts**:
     - Break the transcript into smaller sections if it's too long for the model.
     - Summarize each section and then combine summaries.

2. **Headline Generation with Timestamps**:

   - **Topic Segmentation**:
     - Use algorithms like TextTiling or clustering methods to segment the transcript.
     - **TextTiling Example**:
       ```python
       from nltk.tokenize import TextTilingTokenizer

       tt = TextTilingTokenizer()
       segments = tt.tokenize(transcript_text)
       ```
   - **Keyword Extraction**:
     - Use RAKE or KeyBERT to extract keywords from each segment.
     - **KeyBERT Example**:
       ```python
       from keybert import KeyBERT

       kw_model = KeyBERT()
       keywords = kw_model.extract_keywords(segment_text)
       ```
   - **Generate Headlines**:
     - Use the same summarization model or GPT-3 to generate headlines for each segment.
     - **Example with GPT-3**:
       ```python
       import openai

       openai.api_key = 'YOUR_OPENAI_API_KEY'

       def generate_headline(segment_text):
           prompt = f"Create a concise headline for the following text:\n\n{segment_text}"
           response = openai.Completion.create(
               engine="text-davinci-003",
               prompt=prompt,
               max_tokens=10,
               temperature=0.5,
           )
           return response.choices[0].text.strip()
       ```
   - **Align Headlines with Timestamps**:
     - Record the start and end times of each segment to allow navigation.

3. **Data Preparation for Frontend**:

   - **Structure Data**:
     - Prepare a JSON or dictionary containing summaries, headlines, timestamps, and any other relevant data.
   - **Ensure Compatibility**:
     - Work with the frontend developer to ensure data formats are compatible.

---

### **Team Member 3: Frontend Developer (Streamlit App and Chat Interface)**

**Responsibilities**:

- Develop the Streamlit user interface.
- Integrate backend processing and NLP outputs.
- Implement chat functionality with the video content.

**Tasks**:

1. **Design the Streamlit App Layout**:

   - **Sections to Include**:
     - Video Upload
     - Video Summary Display
     - Interactive Chat Interface
     - Headlines Navigation Panel
   - **User Experience**:
     - Ensure the app is user-friendly and responsive.

2. **Integrate Video Playback**:

   - **Display Video**:
     - Use `st.video()` to display the uploaded video.
   - **Enable Navigation to Timestamps**:
     - Allow users to click on headlines and jump to the corresponding part of the video.
     - **Example**:
       ```python
       if st.button(f"Go to {headline}", key=timestamp):
           st.video(video_file, start_time=timestamp)
       ```

3. **Implement the Chat Interface**:

   - **Chat Functionality**:
     - Use OpenAI's GPT-3 or GPT-4 to enable users to ask questions about the video content.
     - **Example**:
       ```python
       def chat_with_video(user_query, transcript):
           prompt = f"Based on the following transcript, answer the user's question:\n\nTranscript:\n{transcript}\n\nQuestion:\n{user_query}\n\nAnswer:"
           response = openai.Completion.create(
               engine="text-davinci-003",
               prompt=prompt,
               max_tokens=150,
               temperature=0.7,
           )
           return response.choices[0].text.strip()

       user_query = st.text_input("Ask a question about the video:")
       if user_query:
           answer = chat_with_video(user_query, transcript)
           st.write(answer)
       ```
   - **Conversation History**:
     - Maintain a history of the chat for context.

4. **Integrate Summaries and Headlines**:

   - **Display Video Summary**:
     - Show the overall summary generated by the NLP engineer.
   - **Display Headlines with Navigation**:
     - List headlines with clickable links or buttons to jump to video timestamps.

5. **Optimize User Experience**:

   - **Loading Indicators**:
     - Show progress bars or spinners during processing.
   - **Error Handling**:
     - Provide user-friendly error messages for any issues.

---

## **Handling Large Video Files**

- **Chunk Processing**:
  - Divide videos and transcripts into manageable chunks to reduce memory usage.
- **Asynchronous Processing**:
  - Use background processing or Streamlit's caching mechanisms to improve performance.
- **File Compression**:
  - Compress files where possible without losing quality.

---

## **Project Workflow**

1. **User Uploads Video**:
   - Handled by the frontend; triggers backend processing.

2. **Audio Extraction and Transcription**:
   - Backend processes extract audio and transcribe it, handling long videos efficiently.

3. **NLP Processing**:
   - Transcripts are summarized, and headlines are generated with timestamps.

4. **Displaying Results**:
   - Frontend displays the summary, provides chat functionality, and lists headlines for navigation.

5. **User Interaction**:
   - Users can chat with the video content and navigate to points of interest.

---

## **Learning Opportunities**

- **Team Member 1**:
  - Deepen understanding of audio and video processing.
  - Learn about handling large files and optimizing performance.

- **Team Member 2**:
  - Gain experience with NLP models and techniques.
  - Explore summarization, keyword extraction, and topic segmentation.

- **Team Member 3**:
  - Enhance skills in web app development with Streamlit.
  - Learn to integrate AI models and improve user interaction.

---

## **Additional Considerations**

- **API Keys and Costs**:
  - Manage OpenAI API keys securely.
  - Monitor usage to control costs.

- **Data Privacy**:
  - Ensure user-uploaded content is handled securely.
  - Consider adding disclaimers or privacy policies.

- **Testing and Validation**:
  - Test with different video lengths and content types.
  - Validate the accuracy of transcriptions and summaries.

- **Documentation**:
  - Maintain clear documentation for code and user instructions.
  - Use comments and docstrings in your codebase.

---

## **Final Thoughts**

**StreamScribe** offers a comprehensive way to interact with video content, making it an exciting and educational project for your team. By dividing responsibilities and collaborating effectively, you'll create a tool that not only serves practical purposes but also enhances your skills in various aspects of software development and AI.

**Good luck with your project!**

---

**Feel free to reach out if you have any questions or need further assistance with any part of the implementation.**
