import gradio as gr
import whisper
from translate import Translator
from TTS.api import TTS
import uuid
import os
from pathlib import Path
import gc
import torch

os.environ["COQUI_TOS_AGREED"] = "1"

model = whisper.load_model("base")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

output_dir = "output_audio"
os.makedirs(output_dir, exist_ok=True)

def transcribeaudio(audiofile):
    print("Transcribing audio...")
    tresult = model.transcribe(audiofile)

    if "text" not in tresult:
        print("Transcription failed.")
        return {"status": "error", "error": "Transcription failed"}

    audio = whisper.load_audio(audiofile)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    print(f"Detected language: {detected_language}")

    return {"text": tresult["text"], "language": detected_language}

def translatetext(text, source_lang, target_lang):
    try:
        translator = Translator(from_lang=source_lang, to_lang=target_lang)
        translated_text = translator.translate(text)
        print(f"Translated text: {translated_text}")
        return translated_text
    except Exception as e:
        print(f"Error translating to {target_lang}: {str(e)}")
        return f"Error: Could not translate to {target_lang}"

def readtranslation(text, audiofile, language):
    output_path = os.path.join(output_dir, f"{language}_{uuid.uuid4()}.wav")
    print(f"Generating TTS for text: {text}")
    tts.tts_to_file(text=text, file_path=output_path, speaker_wav=audiofile, language=language)
    print(f"Generated audio file at: {output_path}")
    return output_path

def v2vtranslate(audiofile, selected_lang,COQUI_TOS_AGREED, progress=gr.Progress()):

  if COQUI_TOS_AGREED == True:

    progress(0, desc="Starting process...")
    try:
        progress(0.2, desc="Transcribing audio...")
        transcription_result = transcribeaudio(audiofile)

        if isinstance(transcription_result, dict) and transcription_result.get("status") == "error":
            raise gr.Error(transcription_result["error"])

        text = transcription_result["text"]
        detected_language = transcription_result["language"]

        progress(0.4, desc="Translating text...")
        translated_text = translatetext(text, detected_language, selected_lang)

        progress(0.7, desc="Generating audio...")
        audio_path = readtranslation(translated_text, audiofile, selected_lang)

        progress(1.0, desc="Process complete!")
        return audio_path, translated_text
    except Exception as e:
        raise gr.Error(f"An error occurred: {str(e)}")
    finally:
        cleanup_memory()

  else:
        gr.Warning("Please accept the Terms & Condition!")
        return (
            None,
            None,
            None,
            None,
        )

with gr.Blocks() as demo:
    gr.Markdown("## Record yourself in any language and immediately receive voice translations.")

    with gr.Row():
        with gr.Column():
            
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                show_download_button=True,
                max_length=15,
                label="Record your voice",
                waveform_options=gr.WaveformOptions(
                  waveform_color="#01C6FF",
                  waveform_progress_color="#0066B4",
                  skip_length=2,
                  show_controls=False,)
                )
            language_gr = gr.Dropdown(
                label="Language",
                info="Select an output language for the synthesised speech",
                choices=[
                    "en",
                    "es",
                    "fr",
                    "de",
                    "it",
                    "pt",
                    "pl",
                    "tr",
                    "ru",
                    "nl",
                    "cs",
                    "ar",
                    "zh-cn",
                    "ja",
                    "ko",
                    "hu",
                    "hi"
                ],
                max_choices=1,
                value="es",
            )
            tos_gr = gr.Checkbox(
                label="Agree",
                value=False,
                info="I agree to the terms of the CPML: https://coqui.ai/cpml",
            )
            submit = gr.Button("Submit", variant="primary")
            reset = gr.Button("Reset")

    with gr.Row():
        output_audio = gr.Audio(label="Translated Audio", interactive=False)
        output_text = gr.Markdown()

    output_components = [output_audio, output_text]

    submit.click(fn=v2vtranslate, inputs=[audio_input, language_gr,tos_gr], outputs=output_components, show_progress=True)
    reset.click(fn=lambda: None, inputs=None, outputs=output_components + [audio_input])

    def cleanup_memory():
        gc.collect()
        torch.cuda.empty_cache()
        print("Memory cleaned up")

if __name__ == "__main__":
    demo.launch()
