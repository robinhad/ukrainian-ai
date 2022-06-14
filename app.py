import gradio as gr
from transformers import Conversation, ConversationalPipeline, pipeline
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
        with open(file_name, "wb") as file:
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

p = pipeline(
    "automatic-speech-recognition", "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm"
)

conv: ConversationalPipeline = pipeline(
    "conversational", "robinhad/gpt2-uk-conversational"
)

synthesizer = Synthesizer(
    model_path,
    config_path,
    None,
    None,
    None,
)

badge = (
    "https://visitor-badge-reloaded.herokuapp.com/badge?page_id=robinhad.ukrainian-ai"
)


def transcribe(audio, history):
    text = p(audio)["text"]
    history = history or []
    past_user_inputs = [i[0] for i in history]
    generated_responses = [i[1] for i in history]
    response = conv(Conversation(text, past_user_inputs, generated_responses))
    response = response.generated_responses[-1]
    history.append((text, response))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        with torch.no_grad():
            wavs = synthesizer.tts(response)
            synthesizer.save_wav(wavs, fp)
        return text, fp.name, history, history


iface = gr.Interface(
    fn=transcribe,
    inputs=[gr.inputs.Audio(source="microphone", type="filepath"), "state"],
    outputs=[
        gr.outputs.Textbox(label="Recognized text"),
        gr.outputs.Audio(label="Output"),
        gr.outputs.Chatbot(label="Chat"),
        "state",
    ],
    description="""Це альфа-версія end-to-end розмовного бота, з яким можна поспілкуватися голосом.  
    Перейдіть сюди для доступу до текстової версії: [https://huggingface.co/robinhad/gpt2-uk-conversational](https://huggingface.co/robinhad/gpt2-uk-conversational)  
    """,
    article=f"""Розпізнавання української: [https://huggingface.co/Yehor/wav2vec2-xls-r-300m-uk-with-small-lm](https://huggingface.co/Yehor/wav2vec2-xls-r-300m-uk-with-small-lm)  
    Синтез української: [https://huggingface.co/spaces/robinhad/ukrainian-tts](https://huggingface.co/spaces/robinhad/ukrainian-tts)  
    <center><img src="{badge}" alt="visitors badge"/></center>""",
)
iface.launch()
