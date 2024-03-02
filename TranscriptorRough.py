import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"

import pytube
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import moviepy.editor as mp
import speech_recognition as sr
import sqlite3
import requests
from bs4 import BeautifulSoup

conn = sqlite3.connect('prod.db')
cursor = conn.cursor()

# Create a table to store training data if it doesn't exist
def create_table():
    cursor.execute('''CREATE TABLE IF NOT EXISTS training_data
                      (Title, URL, Transcription, [Productivity Label])''')
    conn.commit()

# Insert training data into the database
def insert_data(title, url, transcription, label):
    cursor.execute("INSERT INTO training_data (Title, URL, Transcription, [Productivity Label]) VALUES (?, ?, ?, ?)", (title, url, transcription, label))
    conn.commit()

def get_youtube_video_title(url):
    # Send a GET request to the YouTube video URL
    response = requests.get(url)
    
    # Parse the HTML content of the response
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the title element within the parsed HTML
    title_element = soup.find('title')
    
    # Extract the text of the title element
    video_title = title_element.text if title_element else None
    
    return video_title

# Function to extract features from video content
def extract_features(video_url):
    # Extract video ID from the URL
    video_id = None
    if 'youtube.com/watch?v=' in video_url:
        video_id = video_url.split('youtube.com/watch?v=')[1].split('&')[0]
    elif 'youtu.be/' in video_url:
        video_id = video_url.split('youtu.be/')[1].split('?')[0]

    if video_id is None:
        print("Invalid YouTube URL")
        return None

    # Download the video
    yt = pytube.YouTube("https://www.youtube.com/watch?v=" + video_id)
    stream = yt.streams.first()
    stream.download(filename='video.mp4')

    # Extract text from video using speech recognition
    video = mp.VideoFileClip('video.mp4')
    audio = video.audio
    audio.write_audiofile('audio.wav')
    r = sr.Recognizer()
    
    value = ""  # Initialize with a default value    

## GOOGLE SPEECH RECOGNITION
    # try:
    #     with sr.AudioFile('audio.wav') as source:
    #         audio_data = r.record(source)
    #         value = r.recognize_google(audio_data)

    #     print("Google Text: {}".format(value) + "\n")
       
    # except sr.UnknownValueError:
    #     print("/nOops! Didn't catch that")
        
    # except sr.RequestError as e:
    #     print("/nUh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))

   
## SPHINX SPEECH RECOGNITION
    # try:
    #     with sr.AudioFile('audio.wav') as source:
    #         audio_data = r.record(source)
    #         value = r.recognize_sphinx(audio_data)

    #     print("Sphinx Text: {}".format(value) + "\n")
    
    # except sr.UnknownValueError:
    #     print("Sphinx could not understand audio\n")
    # except sr.RequestError as e:
    #     print("Sphinx error; {0}".format(e) + "\n")

    
## TENSORFLOW SPEECH RECOGNITION
    # try:
    #     with sr.AudioFile('audio.wav') as source:
    #         audio_data = r.record(source)
    #         value = r.recognize_tensorflow(audio_data)

    #     print("Tensorflow Text: {}".format(value) + "\n")
    
    # except sr.UnknownValueError:
    #     print("Tensorflow could not understand audio\n")
    # except sr.RequestError as e:
    #     print("Tensorflow error; {0}".format(e) + "\n")


## WHISPER SPEECH RECOGNITION
    try:
        with sr.AudioFile('audio.wav') as source:
            audio_data = r.record(source)
            value = r.recognize_whisper(audio_data)

        print("\nWhisper Text: {}".format(value) + "\n")
    
    except sr.UnknownValueError:
        print("\nWhisper could not understand audio\n")
    except sr.RequestError as e:
        print("\nWhisper error; {0}".format(e) + "\n")

    return value

# Function to train a model to determine productivity
def train_model(features, labels):
    vectorizer = TfidfVectorizer()
    features_vectorized = vectorizer.fit_transform(features)
    model = LogisticRegression()
    model.fit(features_vectorized, labels)
    return model, vectorizer

# Function to predict productivity of a video
def predict_productivity(model, vectorizer, video_url):
    # Extract features from the video
    text = extract_features(video_url)
    # Vectorize the text using the same vectorizer used during training
    text_vectorized = vectorizer.transform([text])
    # Use the trained model to predict productivity
    prediction = model.predict(text_vectorized)
    return prediction

# Main function
def main():
    create_table()

    # Example dataset
    videos = [
        {"title": get_youtube_video_title("https://www.youtube.com/watch?v=84dYijIpWjQ"), "url": "https://www.youtube.com/watch?v=84dYijIpWjQ", "label": 1},  # productive
        {"title": get_youtube_video_title("https://www.youtube.com/watch?v=IIT29JDuMXs"), "url": "https://www.youtube.com/watch?v=IIT29JDuMXs", "label": 1},  # productive
        {"title": get_youtube_video_title("https://www.youtube.com/watch?v=eIho2S0ZahI"), "url": "https://www.youtube.com/watch?v=eIho2S0ZahI", "label": 1},  # productive
        {"title": get_youtube_video_title("https://www.youtube.com/watch?v=GA7ij5npz-A"), "url": "https://www.youtube.com/watch?v=GA7ij5npz-A", "label": 0},  # un-productive
        {"title": get_youtube_video_title("https://www.youtube.com/watch?v=ywBV6M7VOFU"), "url": "https://www.youtube.com/watch?v=ywBV6M7VOFU", "label": 1},  # productive
        {"title": get_youtube_video_title("https://www.youtube.com/watch?v=Bt48m9fZhmM"), "url": "https://www.youtube.com/watch?v=Bt48m9fZhmM", "label": 0},  # un-productive
        # Add more videos as needed
    ]

    for video in videos:
        # Check if the record already exists in the database
        cursor.execute("SELECT * FROM training_data WHERE URL=?", (video['url'],))
        existing_record = cursor.fetchone()
        if existing_record:
            print("Record already exists for URL:", video['url'])
            continue

        text = extract_features(video['url'])
        insert_data(video['title'], video['url'], text, video['label'])

    # Train your model
    cursor.execute("SELECT Transcription, [Productivity Label] FROM training_data")
    rows = cursor.fetchall()
    features = [row[0] for row in rows]
    labels = [row[1] for row in rows]

    model, vectorizer = train_model(features, labels)

    # Example usage:
    video_url = "https://www.youtube.com/watch?v=9e9D7ABgHpU"
    prediction = predict_productivity(model, vectorizer, video_url)
    if prediction == 1:
        print("Predicted Productivity: Productive")
    else:
        print("Predicted Productivity: Not Productive")

if __name__ == "__main__":
    main()