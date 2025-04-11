from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from gtts import gTTS
import pygame
import os
import tempfile
from PIL import ImageFont, ImageDraw, Image

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("action.h5")

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

# Language toggle (default to English)
selected_language = "English"

sequence = []
sentence = []
predictions = []
threshold = 0.8

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
        print(f"Audio playback error: {e}")

def generate_frames():
    global selected_language
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)

            sequence.append(keypoints)
            sequence[:] = sequence[-30:]

            translated_word = ""

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res) and res[np.argmax(res)] > threshold:
                    predicted_word = actions[np.argmax(res)]
                    translated_word = translations.get(predicted_word, predicted_word) if selected_language == "Hindi" else predicted_word

                    if len(sentence) == 0 or (predicted_word != sentence[-1]):
                        sentence.append(predicted_word)
                        speak(translated_word, 'hi' if selected_language == "Hindi" else 'en')

                if len(sentence) > 5:
                    sentence[:] = sentence[-5:]

            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Convert to PIL for Hindi rendering
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image_pil)
            try:
                font = ImageFont.truetype(hindi_font_path, 32)
                if translated_word and selected_language == "Hindi":
                    draw.text((10, 60), translated_word, font=font, fill=(255, 255, 255))
            except Exception as e:
                print(f"Font error: {e}")
            image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # Overlay recognized sentence
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_language', methods=['POST'])
def set_language():
    global selected_language
    selected_language = request.form.get("language", "English")
    return ("", 204)

if __name__ == '__main__':
    app.run(debug=True)
