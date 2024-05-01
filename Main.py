from flask import Flask, render_template, request
import speech_recognition as sr
import xml.etree.ElementTree as ET
from nltk.tokenize import sent_tokenize, word_tokenize
import soundfile as sf
from nltk import FreqDist
import nltk
from nltk.collocations import*
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
import requests
import pandas as pd
#from aylienapiclient import textapi
import numpy as np
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.cluster.util import cosine_distance
from operator import itemgetter
import urllib
from nltk import FreqDist
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from googletrans import Translator
import os
from pytube import YouTube
import moviepy.editor as mp

app = Flask(__name__)

def download_video(video_url, output_path):
    yt = YouTube(video_url)
    stream = yt.streams.get_highest_resolution()
    stream.download(output_path=output_path)

def download_and_transcribe_latest_video(folder_path):
    transcriptions = {}
    mp4_files = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.mp4')]
    if not mp4_files:
        print("No mp4 files found in the folder.")
        return None
    mp4_files.sort(key=lambda x: os.path.getctime(os.path.join(folder_path, x)), reverse=True)
    latest_video_path = os.path.join(folder_path, mp4_files[0])
    transcriptions = download_and_transcribe_video(latest_video_path)
    return transcriptions

def download_and_transcribe_video(video_path):
    transcriptions = {}
    print(f"Processing video: {video_path}")
    video_clip = mp.VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    audio_clip.write_audiofile(audio_path)
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_duration = audio_clip.duration
        chunk_duration = 200
        num_chunks = int(audio_duration / chunk_duration) + 1
        full_text = ""
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = min((i + 1) * chunk_duration, audio_duration)
            print(f"Processing chunk {i+1}/{num_chunks}")
            chunk = audio_clip.subclip(start_time, end_time)
            chunk_audio_path = os.path.splitext(video_path)[0] + f"chunk{i}.wav"
            chunk.write_audiofile(chunk_audio_path)
            with sr.AudioFile(chunk_audio_path) as chunk_source:
                audio_data = recognizer.record(chunk_source)
                try:
                    text = recognizer.recognize_google(audio_data)
                    print(f"Transcribed text from {os.path.basename(video_path)} chunk {i+1}: {text}")
                    full_text += text
                except sr.UnknownValueError:
                    print("Speech recognition could not understand the audio")
                except sr.RequestError as e:
                    print(f"Could not request results from Google Speech Recognition service; {e}")
        transcriptions[os.path.basename(video_path)] = full_text
    return transcriptions

def translate_to_arabic(text):
    translator = Translator()
    translated_text_chunks = []
    chunk_size = 2000
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        translated_chunk = translator.translate(chunk, dest='ar').text
        translated_text_chunks.append(translated_chunk)
    translated_text = ' '.join(translated_text_chunks)
    return translated_text

def translate_to_french(text):
    translator = Translator()
    translated_text_chunks = []
    chunk_size = 2000
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        translated_chunk = translator.translate(chunk, dest='fr').text
        translated_text_chunks.append(translated_chunk)
    translated_text = ' '.join(translated_text_chunks)
    return translated_text


def summarize_large_text(input_text):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    chunk_size = 1024
    chunks = [input_text[i:i+chunk_size] for i in range(0, len(input_text), chunk_size)]
    summaries = []
    for chunk in chunks:
        inputs = tokenizer([chunk], max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    final_summary = ". ".join(summaries)
    return final_summary

def summarize_large_arabic_text(input_text):
    model_name = "facebook/mbart-large-50"
    tokenizer = MBart50Tokenizer.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    chunk_size = 1024
    chunks = [input_text[i:i+chunk_size] for i in range(0, len(input_text), chunk_size)]
    summaries = []
    for chunk in chunks:
        inputs = tokenizer(chunk, max_length=1024, return_tensors="pt", truncation=True, padding=True)
        summary_ids = model.generate(inputs.input_ids, num_beams=4, max_length=150, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    final_summary = ". ".join(summaries)
    return final_summary

def summarize_large_french_text(input_text):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    chunk_size = 1024
    chunks = [input_text[i:i+chunk_size] for i in range(0, len(input_text), chunk_size)]
    summaries = []
    for chunk in chunks:
        inputs = tokenizer([chunk], max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    final_summary = ". ".join(summaries)
    return final_summary


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    video_url = request.form['video_url']
    output_path = os.path.join(app.root_path, 'downloads')
    download_video(video_url, output_path)
    transcriptions = download_and_transcribe_latest_video(output_path)
    transcribed_text = list(transcriptions.values())[0]
    translated_arabic_text = translate_to_arabic(transcribed_text)
    translated_french_text = translate_to_french(transcribed_text)
    english_summary = summarize_large_text(transcribed_text)
    arabic_summary = summarize_large_arabic_text(translated_arabic_text)
    french_summary = summarize_large_french_text(translated_french_text)
    return render_template('index.html', transcribed_text=transcribed_text, 
                           translated_arabic_text=translated_arabic_text, 
                           translated_french_text=translated_french_text,
                           english_summary=english_summary, 
                           arabic_summary=arabic_summary, 
                           french_summary=french_summary)

if __name__ == '__main__':
    app.run(debug=True)
