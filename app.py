import gradio as gr
from transformers import Conversation, ConversationalPipeline, pipeline, AlbertTokenizerFast
import tempfile
import gradio as gr
from ukrainian_tts.tts import TTS, Voices, Stress
from enum import Enum


tts = TTS(device="cpu") # can try gpu, mps

p = pipeline(
    "automatic-speech-recognition", "robinhad/wav2vec2-xls-r-300m-uk"
)


tokenizer = AlbertTokenizerFast.from_pretrained("robinhad/gpt2-uk-conversational")
conv: ConversationalPipeline = pipeline(
    "conversational", "robinhad/gpt2-uk-conversational", tokenizer=tokenizer
)

class VoiceOption(Enum):
    Olena = "Олена (жіночий) 👩"
    Mykyta = "Микита (чоловічий) 👨"
    Lada = "Лада (жіночий) 👩"
    Dmytro = "Дмитро (чоловічий) 👨"
    Olga = "Ольга (жіночий) 👩"


voice_mapping = {
    VoiceOption.Olena.value: Voices.Olena.value,
    VoiceOption.Mykyta.value: Voices.Mykyta.value,
    VoiceOption.Lada.value: Voices.Lada.value,
    VoiceOption.Dmytro.value: Voices.Dmytro.value,
    VoiceOption.Olga.value: Voices.Olga.value,
}


def transcribe(audio, selected_voice, history):
    text = p(audio)["text"]
    history = history or []
    selected_voice = voice_mapping[selected_voice]
    past_user_inputs = [i[0] for i in history]
    generated_responses = [i[1] for i in history]
    next_output_length = len(tokenizer.encode("".join(generated_responses + past_user_inputs))) + 60
    response = conv(Conversation(text, past_user_inputs, generated_responses), max_length=next_output_length, penalty_alpha=0.6, top_k=4)
    response = response.generated_responses[-1]
    history.append((text, response))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        _, output_text = tts.tts(response, selected_voice, Stress.Dictionary.value, fp)
        return text, fp.name, history, history


iface = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath"), 
        gr.components.Radio(
            label="Голос",
            choices=[option.value for option in VoiceOption],
            value=VoiceOption.Olena.value,
        ),
        "state"],
    outputs=[
        gr.outputs.Textbox(label="Recognized text"),
        gr.outputs.Audio(label="Output", type="filepath"),
        gr.outputs.Chatbot(label="Chat"),
        "state",
    ],
    description="""Це альфа-версія end-to-end розмовного бота, з яким можна поспілкуватися голосом.  
    Перейдіть сюди для доступу до текстової версії: [https://huggingface.co/robinhad/gpt2-uk-conversational](https://huggingface.co/robinhad/gpt2-uk-conversational)  
    """,
    article=f"""Розпізнавання української: [https://huggingface.co/robinhad/wav2vec2-xls-r-300m-uk](https://huggingface.co/robinhad/wav2vec2-xls-r-300m-uk)  
    Синтез української: [https://huggingface.co/spaces/robinhad/ukrainian-tts](https://huggingface.co/spaces/robinhad/ukrainian-tts)""",
)
iface.launch()
