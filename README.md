# DS_StreamScribe

Unlock insights, chat with your videos, and navigate with ease.

# StreamScribe

**StreamScribe** is a Streamlit-powered application that allows users to upload long videos, transcribe the audio, generate summaries, enable interactive chat with the content, and create navigable headlines with timestamps.

## **Features**

1. **Upload Long Videos:** Handle videos up to an hour and a half with efficient processing.
2. **Transcribe Audio:** Convert speech to text using OpenAI's Whisper model.
3. **Summarize Video Content:** Generate concise summaries of the video's content.
4. **Interactive Chat Interface:** Ask questions about the video content.
5. **Headlines with Timestamps:** Extract key points with timings and allow users to jump to specific parts of the video.

## **Setup Instructions**

### **Prerequisites**

- Python 3.11
- Git
- Poetry

### **Installation Steps**

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/shaharTasa/DS_StreamScribe.git
   cd DS_StreamScribe


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
