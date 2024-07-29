import gradio as gr
from transformers import pipeline
from bs4 import BeautifulSoup
import requests
from transformers import BertTokenizer, BertForSequenceClassification, AutoConfig
import torch
from pytubefix import YouTube
from pytubefix.cli import on_progress
import whisper

def get_page_title(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.title.string
    except Exception as e:
        return "Error fetching title"

def analyze_sentiment(url):
    yt = YouTube(url, on_progress_callback = on_progress)
    print(yt.title)
    ys = yt.streams.get_audio_only()
    audio = ys.download(mp3=True)
    model = whisper.load_model("tiny")
    data = model.transcribe(audio,fp16=False)
    text = data["text"]
    print(text)
    model_dir = '.'

    config = AutoConfig.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir, config=config)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation for faster inference
    with torch.no_grad():
        # Make the prediction
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    # Convert predicted class to label (0 or 1)
    label = 'negative' if predicted_class == 0 else 'positive'

    # Debug: Print the logits to understand the output
    print(f"Logits: {logits}")
    print(f"Predicted class: {predicted_class}")

    result = label
    return result

# Define the Gradio interface
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(label="Enter YouTube URL"),
    outputs=gr.Textbox(label="Sentiment Analysis Result"),
    title="YouTube Video Sentiment",
    description="Enter a YouTube video URL to analyze the sentiment of its title.",
)

# Launch the interface
iface.launch()