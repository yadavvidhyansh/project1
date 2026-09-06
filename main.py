import os
import threading
import logging
import time
import requests
import webbrowser
import datetime
from urllib.parse import quote

import pygame
import pyttsx3
import speech_recognition as sr
from gtts import gTTS
from openai import OpenAI
from dotenv import load_dotenv

import musicLibrary
import memory_helper
import detect
from contacts import contacts
from ocr import read_from_camera
from yolo1 import QueryEngine
from ultralytics import YOLO

# =========================================================
# SYSTEM INITIALIZATION
# =========================================================

# Load environment variables
load_dotenv()

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nova.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NovaAssistant")

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

recognizer = sr.Recognizer()
engine = pyttsx3.init() 

# Initialize Vision Engine
vision_engine = QueryEngine()

# Thread-safe speech management
is_speaking = False
speech_lock = threading.Lock()
speech_queue_count = 0
queue_lock = threading.Lock()

def speak(text):
    """Entry point for speaking that starts a background thread."""
    global speech_queue_count
    with queue_lock:
        speech_queue_count += 1
    threading.Thread(target=_speak_thread, args=(text,), daemon=True).start()

def wait_for_speech():
    """Block until current speech is finished."""
    # Tiny delay to allow thread to start
    time.sleep(0.2)
    while speech_queue_count > 0:
        time.sleep(0.1)

def _speak_thread(text):
    """Internal thread for handling TTS without blocking the main loop."""
    global speech_queue_count
    
    with speech_lock:
        logger.info(f"Speaking: {text}")
        
        try:
            tts = gTTS(text)
            temp_file = f'temp_{int(time.time())}.mp3'
            tts.save(temp_file) 

            pygame.mixer.init()
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            pygame.mixer.music.unload()
            if os.path.exists(temp_file):
                os.remove(temp_file) 
        except Exception as e:
            logger.error(f"Speech Error: {e}")
        finally:
            with queue_lock:
                speech_queue_count -= 1

def aiProcess(command):
    logger.info(f"Processing AI command: {command}")
    try:
        # max_retries=0 makes the response instant when quota is empty
        client = OpenAI(api_key=OPENAI_API_KEY, max_retries=0)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a virtual assistant named jarvis skilled in general tasks like Alexa and Google Cloud. Give short responses please"},
                {"role": "user", "content": command}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI Error: {e}")
        return "I'm having trouble connecting to my brain right now."

def get_location():
    try:
        response = requests.get(
            "https://ipinfo.io/json",
            timeout=5
        )
        data = response.json()
        city = data.get("city", "Unknown")
        region = data.get("region", "")
        country = data.get("country", "")
        return f"{city}, {region}, {country}"
    except Exception as e:
        logger.error(f"Location Error: {e}")
        return "Sorry, I could not determine your location."


def get_current_date():
    return datetime.datetime.now().strftime("%A, %d %B %Y")


def get_current_time():
    return datetime.datetime.now().strftime("%I:%M %p")

def processCommand(c):
    c_lower = c.lower().strip()
    logger.info(f"Executing command: {c}")

    # 1. MEMORY & PERSONALIZATION (Priority)
    memory_response = memory_helper.parse_and_remember(c)
    if memory_response:
        speak(memory_response)
        return

    if "what is my name" in c_lower or "who am i" in c_lower:
        name = memory_helper.get_memory("user_name")
        if name:
            speak(f"Your name is {name}.")
        else:
            speak("I don't know your name yet. You can tell me by saying My name is, followed by your name.")
        return

    if "what is my" in c_lower:
        key = c_lower.replace("what is my", "").strip()
        val = memory_helper.get_memory(key)
        if val:
            speak(f"Your {key} is {val}.")
            return

    # 2. WEB & APPS
    if "open google" in c_lower:
        webbrowser.open("https://google.com")
        speak("Opening Google")
    elif "open facebook" in c_lower:
        webbrowser.open("https://facebook.com")
        speak("Opening Facebook")
    elif "open youtube" in c_lower:
        webbrowser.open("https://youtube.com")
        speak("Opening YouTube")
    elif "open linkedin" in c_lower:
        webbrowser.open("https://linkedin.com")
        speak("Opening LinkedIn")
    elif "navigate to" in c_lower:
        destination = c_lower.replace("navigate to", "", 1).strip()
        encoded_destination = quote(destination)
        url = (
            "https://www.google.com/maps/dir/?api=1"
            f"&destination={encoded_destination}"
            "&travelmode=walking"
        )
        speak(f"Starting navigation to {destination}")
        webbrowser.open(url)
    elif c_lower.startswith("play"):
        song = c_lower.split(" ")[1]
        if song in musicLibrary.music:
            link = musicLibrary.music[song]
            webbrowser.open(link)
            speak(f"Playing {song}")
        else:
            speak(f"I couldn't find {song} in your library.")
            
    # 3. CONTACTS
    elif "call" in c_lower:
        name = c_lower.replace("call"," ", 1).strip()
        if name in contacts:
            number = contacts[name]
            speak(f"I found {name.title()}. Would you like me to call them?")
        else:
            speak(f"I couldn't find {name} in your contacts.")

    # 4. VISION & OCR
    elif "read this" in c_lower or "read text" in c_lower:
        speak("Please show the text to the camera.")
        text = read_from_camera()
        print("OCR RESULT:", text)
        speak(text) 

    elif (
        "start pothole detection" in c_lower
        or "road safety mode" in c_lower
        or "detect potholes" in c_lower
    ):
        if vision_engine.camera_running:
            vision_engine.stop_camera()
            time.sleep(1)
        detect.start_pothole_detection(speak)

    elif "stop pothole detection" in c_lower or "stop road safety" in c_lower:
        detect.stop_pothole_detection(speak)

    elif (
        "start object detection" in c_lower
        or "start camera" in c_lower
        or "look around" in c_lower
        or "detect objects" in c_lower
    ):
        if detect.running:
            detect.stop_pothole_detection(speak)
            time.sleep(1)
        status = vision_engine.start_camera(speak_callback=speak)
        speak(status)

    elif "stop object detection" in c_lower or "stop camera" in c_lower:
        status = vision_engine.stop_camera()
        speak(status)

    elif "what do you see" in c_lower or "what's in front" in c_lower:
        if not vision_engine.camera_running:
            speak("I need to start the camera first.")
            vision_engine.start_camera(speak_callback=speak)
            time.sleep(2)
        summary = vision_engine.latest_summary
        answer = vision_engine._create_detection_answer(summary, vision_engine.frame_width, vision_engine.frame_height)
        speak(answer)

    # 5. LOCATION & TIME
    elif "where am i" in c_lower or "location" in c_lower:
        location = get_location()
        speak(f"Your current location is {location}")

    elif "date" in c_lower:
        speak(f"Today is {get_current_date()}")

    elif "time" in c_lower:
        speak(f"The current time is {get_current_time()}")

    # 6. NEWS
    elif "news" in c_lower:
        speak("Getting the latest news.")
        try:
            r = requests.get(
                "https://newsapi.org/v2/top-headlines",
                params={"country": "us", "apiKey": NEWS_API_KEY, "pageSize": 5},
                timeout=10
            )
            if r.status_code == 200:
                articles = r.json().get("articles", [])
                if not articles:
                    speak("No news found.")
                else:
                    for article in articles[:3]:
                        speak(article.get("title"))
            else:
                speak("Sorry, I couldn't get the news.")
        except Exception as e:
            logger.error(f"News Error: {e}")
            speak("There was a problem connecting to the news service.")

    # 7. AI FALLBACK (Only if nothing else matched)
    else:
        output = aiProcess(c)
        speak(output)




if __name__ == "__main__":
    logger.info("Nova Assistant Initializing...")
    speak("Initializing Nova")
    wait_for_speech()

    recognizer = sr.Recognizer()

    # Better recognition settings for speed
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.dynamic_energy_adjustment_damping = 0.15
    recognizer.dynamic_energy_ratio = 1.5
    recognizer.pause_threshold = 0.6
    recognizer.phrase_threshold = 0.3
    recognizer.non_speaking_duration = 0.4

    while True:
        try:
            # Wait for any previous speech to finish before listening for wake word
            wait_for_speech()

            with sr.Microphone() as source:

                # Faster noise adjustment
                recognizer.adjust_for_ambient_noise(
                    source,
                    duration=0.2
                )

                print("Say 'hello'...")
                audio = recognizer.listen(
                    source,
                    timeout=5,
                    phrase_time_limit=4
                )

            try:
                word = recognizer.recognize_google(audio)

                print("You said:", word)

            except sr.UnknownValueError:
                print("Could not understand the audio.")
                continue

            except sr.RequestError as e:
                print("Speech recognition service error:", e)
                continue

            # Wake word
            if word.lower().strip() == "hello":

                user_name = memory_helper.get_memory("user_name")
                if user_name:
                    speak(f"Hello {user_name}. How can I help you?")
                else:
                    speak("Hello. How can I help you?")

                # CRITICAL: Wait for Nova to finish speaking before listening
                wait_for_speech()

                with sr.Microphone() as source:

                    print("Listening for your command...")

                    audio = recognizer.listen(
                        source,
                        timeout=5,
                        phrase_time_limit=8
                    )

                try:
                    command = recognizer.recognize_google(audio)

                    print("Command:", command)

                    processCommand(command)

                except sr.UnknownValueError:
                    speak("Sorry, I didn't understand.")
                    print("Could not understand command.")

                except sr.RequestError as e:
                    print("Speech recognition error:", e)

        except sr.WaitTimeoutError:
            print("No speech detected.")

        except KeyboardInterrupt:
            print("\nNova stopped.")
            break

        except Exception as e:
            print("Error:", e)
