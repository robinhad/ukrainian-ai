---
title: "Ukrainian AI"
emoji: 🇺🇦
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version : 3.16
python_version: 3.9
app_file: app.py
pinned: false
---

# Ukrainian-speaking conversational AI
This is a pet project with aim to provide an end-to-end voice chatbot with ability to listen, speak and make a conversation in Ukrainian.

It's a project with an aim to demonstrate current state-of-the-art speech technologies for Ukrainian language.

Link to speaking demo: [https://huggingface.co/spaces/robinhad/ukrainian-ai](https://huggingface.co/spaces/robinhad/ukrainian-ai)  
Link to text demo: [https://huggingface.co/robinhad/gpt2-uk-conversational](https://huggingface.co/robinhad/gpt2-uk-conversational)
# Technologies used:

- [Wav2Vec2 XLS-R 300M fine-tuned to Ukrainian language](https://huggingface.co/robinhad/wav2vec2-xls-r-300m-uk) for speech recognition.
- [Ukrainian VITS TTS](https://github.com/robinhad/ukrainian-tts) for text-to-speech generation.
- Conversational pipeline (this repository)

TODO: training scripts for conversational pipeline

# How to setup:

1. `pip install -r requirements.txt`
2. `python app.py`
