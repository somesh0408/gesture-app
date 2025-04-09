import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from gtts import gTTS
import pygame
import os
import tempfile
from PIL import ImageFont, ImageDraw, Image
from streamlit.components.v1 import html

# Page config
st.set_page_config(page_title="Real-time Gesture Recognition", layout="wide")
st.title("🤟 Real-time Gesture Recognition with Translation & Voice")

# Load model
try:
    model = tf.keras.models.load_model("action.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Label list (actions)
actions = np.array(['hello', 'thanks', 'no'])

# Translation dictionary
translations = {
    "hello": "नमस्ते",
    "thanks": "धन्यवाद",
    "no": "नहीं"
}

# Font path for Hindi rendering
hindi_font_path = r"C:\\Windows\\Fonts\\Mangal.ttf"

# Mediapipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load HTML UI
with open("templates/index.html", 'r', encoding='utf-8') as f:
    custom_html = f.read()
html(custom_html, height=100)

# Language toggle
language = st.radio("Choose Language for Voice Output", ["English", "Hindi"], horizontal=True)

# Mediapipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh   = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh   = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def speak(text, lang_code):
    try:
        tts = gTTS(text=text, lang=lang_code)
        temp_path = os.path.join(tempfile.gettempdir(), "temp_audio.mp3")
        tts.save(temp_path)
        pygame.mixer.init()
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pass
        pygame.mixer.quit()
        os.remove(temp_path)
    except Exception as e:
        st.warning(f"Audio playback error: {e}")

# Streamlit UI placeholders
frame_placeholder = st.empty()

# Webcam + logic
sequence = []
sentence = []
predictions = []
threshold = 0.8

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        image, results = mediapipe_detection(frame, holistic)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        translated_word = ""

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))

            if np.unique(predictions[-10:])[0] == np.argmax(res) and res[np.argmax(res)] > threshold:
                predicted_word = actions[np.argmax(res)]
                translated_word = translations.get(predicted_word, predicted_word) if language == "Hindi" else predicted_word

                if len(sentence) == 0 or (predicted_word != sentence[-1]):
                    sentence.append(predicted_word)
                    speak(translated_word, 'hi' if language == "Hindi" else 'en')

            if len(sentence) > 5:
                sentence = sentence[-5:]

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        try:
            font = ImageFont.truetype(hindi_font_path, 32)
            if translated_word and language == "Hindi":
                draw.text((10, 60), translated_word, font=font, fill=(255, 255, 255))
        except Exception as e:
            st.warning(f"Font error: {e}")
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        frame_placeholder.image(image, channels="BGR")

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()