import os
import cv2
import numpy as np
import logging
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
import uvicorn
from dotenv import load_dotenv

# Import Nova core modules
import memory_helper
from yolo1 import QueryEngine
import detect
from ocr import reader
import requests
from openai import OpenAI

# Initialize
load_dotenv()
app = FastAPI(title="Nova AI Backend")

# We'll use a single engine for general detection 
vision_engine = QueryEngine()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

class CommandRequest(BaseModel):
    text: str

class CommandResponse(BaseModel):
    response: str
    action: Optional[str] = None

@app.get("/status")
def get_status():
    return {"status": "online", "name": "Nova AI", "version": "2.0"}

@app.post("/command", response_model=CommandResponse)
async def process_command(request: CommandRequest):
    text = request.text
    
    # Check Memory
    memory_response = memory_helper.parse_and_remember(text)
    if memory_response:
        return CommandResponse(response=memory_response)

    # Who am I check
    if "what is my name" in text.lower() or "who am i" in text.lower():
        name = memory_helper.get_memory("user_name")
        if name:
            return CommandResponse(response=f"Your name is {name}.")
        else:
            return CommandResponse(response="I don't know your name yet.")

    # Pothole Check
    if "pothole" in text.lower() or "road safety" in text.lower():
        return CommandResponse(response="Activating road safety analysis...", action="ROAD_SAFETY")

    # News check
    if "news" in text.lower():
        return CommandResponse(response="Getting the latest news...", action="GET_NEWS")

    # Fallback to AI
    try:
        client = OpenAI(api_key=OPENAI_API_KEY, max_retries=0)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a virtual assistant named jarvis. Give short responses."},
                {"role": "user", "content": text}
            ]
        )
        return CommandResponse(response=completion.choices[0].message.content)
    except Exception as e:
        return CommandResponse(response="I'm having trouble connecting to my brain.")

@app.post("/vision")
async def process_vision(file: UploadFile = File(...), mode: str = "general"):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if mode == "pothole":
        try:
            # Dynamically load pothole model if not loaded
            pothole_model = detect.load_model()
            results = pothole_model.predict(img, conf=0.65, verbose=False)
            
            potholes = []
            for result in results:
                for box in result.boxes:
                    potholes.append("pothole")
            
            if potholes:
                return {"response": f"Warning! I detected {len(potholes)} potholes on the road."}
            else:
                return {"response": "The road ahead looks clear."}
        except Exception as e:
            return {"response": "I couldn't run road safety check right now."}

    # Run General YOLO
    result = vision_engine.detector.detect(img)
    summary = vision_engine.detector.get_detection_summary(result, detect_color=True)
    answer = vision_engine._create_detection_answer(summary, img.shape[1], img.shape[0])
    return {"response": answer, "summary": summary}

@app.post("/ocr")
async def process_ocr(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = reader.readtext(img)
    texts = [res[1] for res in results if res[2] > 0.40]
    
    final_text = " ".join(texts) if texts else "I couldn't find any text."
    return {"response": final_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
