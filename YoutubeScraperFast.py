from pytube import YouTube
import tensorflow as tf
import numpy as np
import cv2
import imageio
from concurrent.futures import ThreadPoolExecutor
import os
import sqlite3
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json

# Create a database connection
conn = sqlite3.connect('fingerprints.db')
c = conn.cursor()
if conn is None:
    print("Error! cannot create the database connection.")

# Create table
c.execute('''CREATE TABLE IF NOT EXISTS fingerprints
             (video_url text, title text, length integer, views integer, rating integer, thumbnail_url text, description text, watch_url text, features blob)''')

# Initialize the model for feature extraction
model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet')

def video_Info(yt):
    info = {
        "title": yt.title,
        "length": yt.length,
        "views": yt.views,
        "age_restricted": yt.age_restricted,
        "rating": yt.rating,
        "thumbnail_url": yt.thumbnail_url,
        "description": yt.description,
        "watch_url": yt.watch_url
    }
    for key, value in info.items():
        print(f"{key} : {value}")
    return info

# Extract 5-second clips from a video
# Extract 5-second clips from a video
def extract_segments(video_path, segment_length=1, frame_skip=10):
    reader = imageio.get_reader(video_path, format='ffmpeg')
    print(reader.get_meta_data())
    fps = reader.get_meta_data()['fps']
    duration = reader.get_meta_data()['duration']
    total_frames = int(fps * duration)
    
    segments = []
    for start in range(0, total_frames, int(fps * segment_length * frame_skip)):
        end = min(int(start + fps * segment_length), total_frames)
        segments.append((start, end))
    reader.close()
    return segments

# Extract and process frames from a segment
def process_segment(video_path, segment, fps=1, batch_size=16):
    start_frame, end_frame = segment
    reader = imageio.get_reader(video_path)
    frames = []
    features = []
    
    for frame_idx in range(start_frame, end_frame):
        try:
            frame = reader.get_data(frame_idx)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame)
        except IndexError:
            print(f"Frame {frame_idx} not found.")
            break
        except StopIteration:
            print("Reached end of video.")
            break
        
        if len(frames) == batch_size:
            batch_features = [extract_features(frame) for frame in frames]
            features.extend(batch_features)
            frames = []
    
    if frames:  # Process remaining frames
        batch_features = [extract_features(frame) for frame in frames]
        features.extend(batch_features)
    
    reader.close()
    return features

# Extract features from a frame
def extract_features(frame):
    frame = cv2.resize(frame, (128, 128)).astype('float32')
    frame = tf.keras.applications.mobilenet_v2.preprocess_input(frame)
    features = model.predict(np.expand_dims(frame, axis=0))
    return features

# Main routine
def process_video(video_path):
    segments = extract_segments(video_path)
    aggregated_features = []
    
    for segment in segments:
        features = process_segment(video_path, segment)
        # Aggregate features (for example, by averaging)
        if features:
            aggregated_feature = np.mean(features, axis=0)  # Taking average, adjust as needed
            aggregated_features.append(aggregated_feature)
        
    return aggregated_features

# Loop through all the words in words.txt
with open('words.txt') as f:
    for line in f:
        # Remove lines that are blank
        if line == '\n':
            continue

        # Pull random video from YouTube
        driver = webdriver.Chrome()
        previous_num_of_videos = 0
        wait = WebDriverWait(driver, 4)
        # Open YouTube
        driver.get('https://www.youtube.com/results?search_query='+line)
        # Wait for page to load
        driver.implicitly_wait(10)
        # Scroll down to load more videos
        while True:
            driver.execute_script("window.scrollBy(0, 50000)")
            
            try:
                # Waiting for the loading symbol to disappear
                wait.until(EC.invisibility_of_element_located((By.XPATH, 'xpath_of_loading_icon')))
            except Exception as e:
                print(e)
                break
            
            video_links = driver.find_elements(By.ID, 'video-title')
            
            # If the number of videos did not increase after scrolling, we're probably at the end.
            if len(video_links) == previous_num_of_videos:
                break
            previous_num_of_videos = len(video_links)
        # Get all the video links
        video_links = driver.find_elements('id', 'video-title')
        video_url = []
        for link in video_links:
            url = link.get_attribute('href')
            if url and 'youtube.com' in url:
                video_url.append(url)
            elif url:
                video_url.append("https://youtube.com" + url)

        print(f"Found {len(video_url)} videos")
        for url in video_url:
            print(f"Processing {url}")
            print(f"I have {len(video_url)} videos left")
            yt = YouTube(url)
            if yt is None:
                print("Video not found")
                continue
            try:
                if yt.length > 1300:
                    print("Video is too long")
                    continue
            except TypeError:
                print("Video is too long")
                continue

            try:
                video = yt.streams.filter(file_extension='mp4').first()
                video_file = video.download()
            except:
                print("Live video")
                continue
            # Check if the video is already in the database
            c.execute("SELECT * FROM fingerprints WHERE video_url=?", (url,))
            if c.fetchone() is not None:
                print("Video already in database")
                os.remove(video_file)
                continue
            features = process_video(video_file)
            # Convert the list of NumPy arrays to a list of lists
            features_list = [feature.tolist() for feature in features]
            # Convert the list of lists to a JSON string
            features_json = json.dumps(features_list)
            # Convert JSON string to bytes
            features_bytes = features_json.encode('utf-8')
            # Insert a row of data
            video_info = video_Info(yt)
            c.execute("INSERT INTO fingerprints VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                        (url, 
                        video_info['title'], 
                        video_info['length'], 
                        video_info['views'], 
                        video_info['rating'], 
                        video_info['thumbnail_url'], 
                        video_info['description'], 
                        video_info['watch_url'], 
                        features_bytes))
            conn.commit()
            os.remove(video_file)