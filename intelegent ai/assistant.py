import os
# Suppress warnings early
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import cmd
import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re
from datetime import datetime
import speech_recognition as sr
import pyttsx3
import cv2
from pyngrok import ngrok, conf
import threading
import time
import logging
from transformers import pipeline
import http.server
import socketserver
import json
import torch
import warnings

# Suppress additional warnings
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler('assistant.log'),
    logging.StreamHandler()
])

# Download NLTK data
nltk.download('punkt', quiet=True)

# Configuration
CONFIG = {
    'NGROK_AUTH_TOKEN': os.getenv('NGROK_AUTH_TOKEN', 'YOUR_NGROK_AUTH_TOKEN'),
    'VOICE_ID': 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_DAVID_11.0',
    'MODEL_RETRAIN_INTERVAL': 10,
    'IDLE_TIMEOUT': 600,  # 10 minutes
    'QUESTION_TIMEOUT': 300  # 5 minutes
}

# Check for GPU and CUDA
def check_gpu():
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')
    logging.info(f"Using device: {device}")
    if cuda_available:
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA Version: {torch.version.cuda}")
    else:
        logging.warning("CUDA not available. Using CPU. Ensure CUDA 12.1 and cuDNN are installed.")
    return device

device = check_gpu()

# Initialize SQLite database
def init_db():
    try:
        conn = sqlite3.connect('assistant.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS interactions
                     (id INTEGER PRIMARY KEY, input_type TEXT, input_text TEXT, response TEXT, feedback TEXT, timestamp TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS trading_ideas
                     (id INTEGER PRIMARY KEY, idea TEXT, timestamp TEXT)''')
        conn.commit()
        conn.close()
        logging.info("Database initialized.")
    except Exception as e:
        logging.error(f"Database initialization error: {e}")
        raise

# Store interaction
def store_interaction(input_type, input_text, response, feedback=None):
    try:
        conn = sqlite3.connect('assistant.db')
        c = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute("INSERT INTO interactions (input_type, input_text, response, feedback, timestamp) VALUES (?, ?, ?, ?, ?)",
                  (input_type, input_text, response, feedback, timestamp))
        conn.commit()
        conn.close()
        logging.info(f"Stored interaction: {input_type}, {input_text}")
    except Exception as e:
        logging.error(f"Store interaction error: {e}")

# Store trading idea
def store_trading_idea(idea):
    try:
        conn = sqlite3.connect('assistant.db')
        c = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute("INSERT INTO trading_ideas (idea, timestamp) VALUES (?, ?)", (idea, timestamp))
        conn.commit()
        conn.close()
        logging.info(f"Stored trading idea: {idea}")
    except Exception as e:
        logging.error(f"Store trading idea error: {e}")

# Load interaction data
def load_data():
    try:
        conn = sqlite3.connect('assistant.db')
        df = pd.read_sql_query("SELECT input_text, response FROM interactions WHERE feedback IS NOT NULL", conn)
        conn.close()
        return df
    except Exception as e:
        logging.error(f"Load data error: {e}")
        return pd.DataFrame()

# Train model
def train_model():
    try:
        df = load_data()
        if len(df) < 2:
            logging.info("Not enough data to train model.")
            return None, None
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df['input_text'])
        y = df['response']
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)
        logging.info("Model retrained.")
        return model, vectorizer
    except Exception as e:
        logging.error(f"Train model error: {e}")
        return None, None

# Get recent context (increased limit)
def get_recent_context(limit=15):
    try:
        conn = sqlite3.connect('assistant.db')
        c = conn.cursor()
        c.execute("SELECT input_text, response FROM interactions ORDER BY timestamp DESC LIMIT ?", (limit,))
        context = c.fetchall()
        conn.close()
        return context
    except Exception as e:
        logging.error(f"Get context error: {e}")
        return []

# Process input with enhanced context
def process_input(input_type, input_text, model, vectorizer, generator):
    try:
        input_text = input_text.lower().strip()
        context = get_recent_context()
        context_text = "\n".join([f"User: {inp}\nAssistant: {resp}" for inp, resp in context])
        prompt = f"{context_text}\nUser: {input_text}\nAssistant:"

        if model is None or vectorizer is None:
            if 'remind' in input_text:
                response = "Reminder set! What time should I remind you?"
            elif 'suggest' in input_text or input_text == 'wave':
                response = "How about trying a new project or reviewing your goals?"
            elif input_text == 'thumbs-up':
                response = "Awesome! Want to set a new task or goal?"
            elif 'trad' in input_text or 'bot' in input_text:
                response = "Let’s dive into trading! Want to brainstorm a new strategy or tweak your bot?"
            elif input_text in ['hey', 'hi', 'hello']:
                response = "Hey! What’s on your mind today? A project, trading, or something else?"
            elif input_text in ['yes', 'yeah', 'yup']:
                response = "Great! What do you want to explore next?"
            else:
                generated = generator(prompt, max_length=200, num_return_sequences=1, pad_token_id=50256, truncation=True)[0]['generated_text']
                response = generated.split("Assistant:")[-1].strip()
                if not response:
                    response = "I’m not sure, but let’s chat! What’s up?"
        else:
            X = vectorizer.transform([input_text])
            response = model.predict(X)[0]
        return response
    except Exception as e:
        logging.error(f"Process input error: {e}")
        return "Sorry, something went wrong. Try again."

# Generate context-aware question
def ask_question():
    try:
        classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', revision='714eb0f', device=0 if device.type == 'cuda' else -1)
        conn = sqlite3.connect('assistant.db')
        c = conn.cursor()
        c.execute("SELECT input_text, response, timestamp FROM interactions ORDER BY timestamp DESC LIMIT 5")
        recent_interactions = c.fetchall()
        conn.close()
        
        if recent_interactions:
            last_input, last_response, last_time = recent_interactions[0]
            last_time = datetime.strptime(last_time, '%Y-%m-%d %H:%M:%S')
            sentiment = classifier(last_input)[0]['label']
            
            trading_context = any('trad' in interaction[0].lower() or 'bot' in interaction[0].lower() for interaction in recent_interactions)
            
            if (datetime.now() - last_time).total_seconds() > CONFIG['QUESTION_TIMEOUT']:
                return "Been a while! What’s something new you want to explore?"
            elif trading_context:
                return "Still thinking about trading? Want to discuss a new strategy or review your bot?"
            elif sentiment == 'POSITIVE':
                return "You seem upbeat! What’s inspiring you today?"
            elif sentiment == 'NEGATIVE':
                return "Sounds like something’s on your mind. Want to talk about it?"
            else:
                context = "\n".join([f"User: {inp}\nAssistant: {resp}" for inp, resp in recent_interactions])
                prompt = f"{context}\nAssistant: What should we talk about next?"
                question = generator(prompt, max_length=120, num_return_sequences=1, pad_token_id=50256, truncation=True)[0]['generated_text']
                question = question.split("Assistant:")[-1].strip()
                return question or "What’s next on your mind? A project, trading, or maybe something new?"
        return "What’s on your mind today? A project, idea, or maybe trading?"
    except Exception as e:
        logging.error(f"Question generation error: {e}")
        return "What’s up? Got any ideas to share?"

# Initialize TTS
def init_tts():
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        voice_set = False
        for voice in voices:
            if CONFIG['VOICE_ID'] in voice.id:
                engine.setProperty('voice', voice.id)
                voice_set = True
                break
        if not voice_set:
            engine.setProperty('voice', voices[0].id)
        engine.setProperty('rate', 150)
        return engine
    except Exception as e:
        logging.error(f"TTS initialization error: {e}")
        return None

# Recognize speech
def recognize_speech():
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)
        text = recognizer.recognize_google(audio)
        logging.info(f"Recognized speech: {text}")
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the audio."
    except sr.RequestError:
        return "Speech recognition service is unavailable."
    except Exception as e:
        logging.error(f"Speech recognition error: {e}")
        return "Error processing audio."

# Detect gesture
def detect_gesture():
    try:
        cascade_path = 'haarcascade_hand.xml'
        if not os.path.exists(cascade_path):
            logging.warning("Haar cascade file missing. Using threshold detection.")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return "Webcam not accessible."
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return "Failed to capture image."
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cap.release()
            if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > 1000:
                return "wave"
            return None
        cascade = cv2.CascadeClassifier(cascade_path)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Webcam not accessible."
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return "Failed to capture image."
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hands = cascade.detectMultiScale(gray, 1.1, 5)
        cap.release()
        if len(hands) > 0:
            x, y, w, h = hands[0]
            if w * h > 1000:
                return "wave"
            elif w * h > 500:
                return "thumbs-up"
        return None
    except Exception as e:
        logging.error(f"Gesture detection error: {e}")
        return "Error detecting gesture."

# HTTP server
class PublicAPIHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        try:
            if self.path == '/public':
                conn = sqlite3.connect('assistant.db')
                c = conn.cursor()
                c.execute("SELECT idea, timestamp FROM trading_ideas ORDER BY timestamp DESC LIMIT 5")
                ideas = c.fetchall()
                conn.close()
                response = {
                    'status': 'Assistant is online.',
                    'recent_ideas': [{'idea': idea, 'timestamp': ts} for idea, ts in ideas]
                }
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_response(404)
                self.end_headers()
        except Exception as e:
            logging.error(f"Public GET error: {e}")
            self.send_response(500)
            self.end_headers()

    def do_POST(self):
        try:
            if self.path == '/public' or self.path == '/public/trading':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data)
                idea = data.get('idea', '')
                if idea:
                    store_trading_idea(idea)
                    response = {'status': f'Idea recorded: {idea}'}
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())
                else:
                    self.send_response(400)
                    self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()
        except Exception as e:
            logging.error(f"Public POST error: {e}")
            self.send_response(500)
            self.end_headers()

# Monitor activity
def monitor_activity():
    public_tunnel = None
    server = None
    retries = 3
    ngrok_enabled = True

    if CONFIG['NGROK_AUTH_TOKEN'] == 'YOUR_NGROK_AUTH_TOKEN':
        logging.error("Invalid ngrok authtoken. Set a valid token in NGROK_AUTH_TOKEN or CONFIG. Disabling ngrok.")
        ngrok_enabled = False
        print("Ngrok disabled due to invalid authtoken. Assistant will run without public API.")

    while True:
        try:
            conn = sqlite3.connect('assistant.db')
            c = conn.cursor()
            c.execute("SELECT timestamp FROM interactions ORDER BY timestamp DESC LIMIT 1")
            last_interaction = c.fetchone()
            conn.close()
            if last_interaction:
                last_time = datetime.strptime(last_interaction[0], '%Y-%m-%d %H:%M:%S')
                if (datetime.now() - last_time).total_seconds() > CONFIG['IDLE_TIMEOUT'] and ngrok_enabled:
                    if not public_tunnel:
                        for attempt in range(retries):
                            try:
                                conf.get_default().auth_token = CONFIG['NGROK_AUTH_TOKEN']
                                public_tunnel = ngrok.connect(8000)
                                logging.info(f"Public URL: {public_tunnel.public_url}")
                                handler = PublicAPIHandler
                                server = socketserver.TCPServer(('', 8000), handler)
                                threading.Thread(target=server.serve_forever, daemon=True).start()
                                logging.info("HTTP server started on port 8000")
                                break
                            except Exception as e:
                                logging.error(f"Ngrok retry {attempt + 1}/{retries} failed: {e}")
                                if attempt < retries - 1:
                                    time.sleep(5)
                                else:
                                    logging.error("Ngrok failed after retries. Disabling ngrok.")
                                    ngrok_enabled = False
                else:
                    if public_tunnel:
                        ngrok.disconnect(public_tunnel.public_url)
                        public_tunnel = None
                        if server:
                            server.shutdown()
                            server.server_close()
                            server = None
                            logging.info("HTTP server and ngrok tunnel closed.")
            time.sleep(60)
        except Exception as e:
            logging.error(f"Monitor activity error: {e}")
            time.sleep(60)

# CLI Assistant
class AssistantCLI(cmd.Cmd):
    prompt = 'Assistant> '
    intro = """
Welcome to the Intelligent Assistant!
I'm here to help with projects, ideas, trading, or anything else on your mind.
Available commands:
  text <message>  : Send a text message (or just type the message directly)
  t <message>     : Alias for text
  audio           : Record audio input
  visual          : Capture a webcam gesture
  feedback <good/bad> : Rate the last response
  trading <idea>  : Log a trading idea
  list_trading    : View recent trading ideas
  quit            : Exit
Examples:
  text What's on your mind?
  t suggest a project
  trading New EUR/USD strategy
  hey (direct input works!)
Type 'help' for details. To answer questions, use 'text <answer>' or just type the answer.
"""

    def __init__(self):
        super().__init__()
        self.model, self.vectorizer = train_model()
        self.engine = init_tts()
        self.generator = pipeline('text-generation', model='gpt2', device=0 if device.type == 'cuda' else -1)
        self.interaction_count = 0
        self.last_input = ''
        self.last_response = ''
        self.last_input_type = 'text'

    def default(self, line):
        self.do_text(line)

    def do_text(self, arg):
        if not arg:
            print("Please provide a message.")
            return
        self.last_input = arg
        self.last_input_type = 'text'
        response = process_input('text', arg, self.model, self.vectorizer, self.generator)
        self.last_response = response
        store_interaction('text', arg, response)
        print(response)
        if self.engine:
            self.engine.say(response)
            self.engine.runAndWait()
        self.interaction_count += 1
        if self.interaction_count % CONFIG['MODEL_RETRAIN_INTERVAL'] == 0:
            self.model, self.vectorizer = train_model()
            print("(Model retrained)")

    def do_t(self, arg):
        self.do_text(arg)

    def do_audio(self, arg):
        input_text = recognize_speech()
        self.last_input = input_text
        self.last_input_type = 'audio'
        print(f"You said: {input_text}")
        response = process_input('audio', input_text, self.model, self.vectorizer, self.generator)
        self.last_response = response
        store_interaction('audio', input_text, response)
        print(response)
        if self.engine:
            self.engine.say(response)
            self.engine.runAndWait()
        self.interaction_count += 1
        if self.interaction_count % CONFIG['MODEL_RETRAIN_INTERVAL'] == 0:
            self.model, self.vectorizer = train_model()
            print("(Model retrained)")

    def do_visual(self, arg):
        input_text = detect_gesture()
        self.last_input = input_text
        self.last_input_type = 'visual'
        if not input_text:
            print("No gesture detected.")
            return
        print(f"Detected: {input_text}")
        response = process_input('visual', input_text, self.model, self.vectorizer, self.generator)
        self.last_response = response
        store_interaction('visual', input_text, response)
        print(response)
        if self.engine:
            self.engine.say(response)
            self.engine.runAndWait()
        self.interaction_count += 1
        if self.interaction_count % CONFIG['MODEL_RETRAIN_INTERVAL'] == 0:
            self.model, self.vectorizer = train_model()
            print("(Model retrained)")

    def do_feedback(self, arg):
        if arg.lower() not in ['good', 'bad']:
            print("Please specify 'good' or 'bad'.")
            return
        store_interaction(self.last_input_type, self.last_input, self.last_response, arg.lower())
        self.model, self.vectorizer = train_model()
        print(f"Feedback recorded: {arg.lower()}")
        if self.engine:
            self.engine.say(f"Feedback recorded: {arg.lower()}")
            self.engine.runAndWait()

    def do_trading(self, arg):
        if not arg:
            print("Please provide a trading idea.")
            return
        store_trading_idea(arg)
        print(f"Trading idea recorded: {arg}")
        if self.engine:
            self.engine.say(f"Trading idea recorded: {arg}")
            self.engine.runAndWait()

    def do_list_trading(self, arg):
        try:
            conn = sqlite3.connect('assistant.db')
            c = conn.cursor()
            c.execute("SELECT idea, timestamp FROM trading_ideas ORDER BY timestamp DESC LIMIT 5")
            ideas = c.fetchall()
            conn.close()
            if ideas:
                print("Recent trading ideas:")
                for idea, timestamp in ideas:
                    print(f"- {idea} ({timestamp})")
            else:
                print("No trading ideas yet.")
        except Exception as e:
            logging.error(f"List trading error: {e}")
            print("Error listing trading ideas.")

    def do_quit(self, arg):
        print("Goodbye!")
        return True

    def postcmd(self, stop, line):
        question = ask_question()
        print(f"Question: {question}")
        print("(Use 'text <answer>' or just type the answer)")
        if self.engine:
            self.engine.say(question)
            self.engine.runAndWait()
        return stop

if __name__ == '__main__':
    init_db()
    threading.Thread(target=monitor_activity, daemon=True).start()
    AssistantCLI().cmdloop()