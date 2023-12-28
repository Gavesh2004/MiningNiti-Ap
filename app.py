import os
import sre_compile
import sre_constants
from gtts import gTTS
import yaml
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn


from pydantic import BaseModel

from src.model import MyChatBot

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

with open("conf/variables.yaml", 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Initialize an instance of MyChatBot and set the prompt
model = MyChatBot(api_key=api_key, temperature=config['temperature'])
model.set_prompt(template=config['MiningNitiTemplate'], input_variables=["input"])

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3000/chatting",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],  # Adjust this line based on the headers your frontend sends
)

class Item(BaseModel):
    input_query: str

@app.post("/speech-to-text")
def speech_to_text(request: Request):
    audio_file = request.json['audioFile']
    r = sre_compile.Recognizer()
    with sre_constants.AudioFile(audio_file) as source:
        audio_text = r.listen(source)
        try:
            text = r.recognize_google(audio_text)
            return {"transcript": text}
        except:
            return {"error": "Error transcribing audio"}

@app.post("/text-to-speech")
def text_to_speech(request: Request):
    text = request.json['text']
    language = 'en'
    myobj = gTTS(text=text, lang=language, slow=False)
    myobj.save("welcome.mp3")
    return JSONResponse(content={"status": "success"})

@app.post("/chat")
def run(item: Item, model: MyChatBot = Depends(lambda: model)):
    return model.run(input=item.input_query)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 
