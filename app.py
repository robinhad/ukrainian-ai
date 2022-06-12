import random
import gradio as gr
from transformers import pipeline
import tempfile
import torch
from os.path import exists
import requests
from TTS.utils.synthesizer import Synthesizer
import gradio as gr

def download(url, file_name):
    if not exists(file_name):
        print(f"Downloading {file_name}")
        r = requests.get(url, allow_redirects=True)
        with open(file_name, 'wb') as file:
            file.write(r.content)
    else:
        print(f"Found {file_name}. Skipping download...")


print("downloading uk/mykyta/vits-tts")
release_number = "v2.0.0-beta"
model_link = f"https://github.com/robinhad/ukrainian-tts/releases/download/{release_number}/model-inference.pth"
config_link = f"https://github.com/robinhad/ukrainian-tts/releases/download/{release_number}/config.json"

model_path = "model.pth"
config_path = "config.json"

download(model_link, model_path)
download(config_link, config_path)

p = pipeline("automatic-speech-recognition", "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm")

synthesizer = Synthesizer(
    model_path, config_path, None, None, None,
)

badge = "https://visitor-badge-reloaded.herokuapp.com/badge?page_id=robinhad.ukrainian-ai"

def transcribe(audio):
    text = p(audio)["text"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        with torch.no_grad():
            wavs = synthesizer.tts(text)
            synthesizer.save_wav(wavs, fp)
        return fp.name

gr.Interface(
    fn=transcribe, 
    inputs=gr.inputs.Audio(source="microphone", type="filepath"), 
    outputs=gr.outputs.Audio(label="Output"),
    article=f"<center><img src=\"{badge}\" alt=\"visitors badge\"/></center>",).launch()

def chat(message, history):
    history = history or []
    #if message.startswith("How many"):
    #    response = random.randint(1, 10)
    #elif message.startswith("How"):
    #    response = random.choice(["Great", "Good", "Okay", "Bad"])
    #elif message.startswith("Where"):
    #    response = random.choice(["Here", "There", "Somewhere"])
    #else:
    #    response = "I don't know"
    #history.append((message, response))
    return history, history


#iface = gr.Interface(
#    chat,
#    ["audio", "state"],
#    ["chatbot", "state"],
#    allow_screenshot=False,
#    allow_flagging="never",
#)
#iface.launch()