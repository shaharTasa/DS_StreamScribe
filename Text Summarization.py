from transformers import pipeline
import json
import nltk
import os
import requests
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from pydantic import BaseModel, Field
from deep_translator import GoogleTranslator
import string

data ={
    "segments": [
        {
            "start_time": "00:00:10",
            "end_time": "00:01:00",
            "summary": "Summary of the first segment",
            "headline": "Headline for the first segment",
            "keywords": ["keyword1", "keyword2"]
        },
        {
            "start_time": "00:01:01",
            "end_time": "00:02:00",
            "summary": "Summary of the second segment",
            "headline": "Headline for the second segment",
            "keywords": ["keyword3", "keyword4"]
        }
    ]
}


json_data= json.dumps(data, indent=4)

with open("data.json", "w") as f:
    f.write(json_data)


nltk.download('punkt')

def split_text(text, max_tokens=512):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= max_tokens:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    chunks = split_text(text)
    summaries = []

    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    # Combine the summaries into one coherent text
    combined_summary = " ".join(summaries)
    return combined_summary


text = "Your very long transcript goes here..."
summary = summarize_text(text)
print(summary)

#chat bot
GROQ_API_KEY = os.environ["GROQ_API_KEY"] = "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"

class Search(BaseModel):
    """Class for generating answer for user question"""
    setup: str = Field(..., text=text)
    question: str = Field (...)
    answer: str = Field(..., description="answer")

# Initialize the language model
llm = ChatGroq(
    model="llama3-groq-70b-8192-tool-use-preview",
    api_key=GROQ_API_KEY
)

# Updated goods string with a focus on related products
answer = (
    "answer the question the user inputs about the text"
)

# Update prompt to better specify related goods
prompt = ChatPromptTemplate(
    [
        SystemMessagePromptTemplate.from_template(
            "You are an expert in the transcription extracted from the {text}. answer the question according to transcription ."
        ),
        HumanMessagePromptTemplate.from_template(
            "Here is the transcription text:\n\n{text}\n\nBased on the transcription, please answer the following question:\n\n{question}"
        )
    ]
)

# Constructing the chain
chain = prompt | llm.with_structured_output(Search)