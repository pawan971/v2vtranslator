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

output_dir = "/content/output_audio"
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

def translatetext(text, source_lang):
    translations = {}
    languages = {"es": "Spanish", "fr": "French", "hi": "Hindi"}
    
    for lang_code, lang_name in languages.items():
        try:
            translator = Translator(from_lang=source_lang, to_lang=lang_code)
            translated_text = translator.translate(text)
            translations[lang_code] = translated_text
            print(f"{lang_name} Translation: {translated_text}")
        except Exception as e:
            print(f"Error translating to {lang_name}: {str(e)}")
            translations[lang_code] = f"Error: Could not translate to {lang_name}"
    
    return [translations[lang] for lang in ["es", "fr", "hi"]]

def readtranslation(text, audiofile, language):
    output_path = os.path.join(output_dir, f"{language}_{uuid.uuid4()}.wav")
    print(f"Generating TTS for text: {text}")
    tts.tts_to_file(text=text,
                    file_path=output_path,
                    speaker_wav=audiofile,
                    language=language)
    print(f"Generated audio file at: {output_path}")
    return output_path

def voice_to_voice(audiofile, progress=gr.Progress()):
    progress(0, desc="Starting process...")
    try:
        progress(0.2, desc="Transcribing audio...")
        transcription_result = transcribeaudio(audiofile)
        
        if isinstance(transcription_result, dict) and transcription_result.get("status") == "error":
            raise gr.Error(transcription_result["error"])
        
        text = transcription_result["text"]
        detected_language = transcription_result["language"]
        
        progress(0.4, desc="Translating text...")
        translations = translatetext(text, detected_language)
        
        audio_paths = []
        languages = ["es", "fr", "hi"]
        for i, (lang, translation) in enumerate(zip(languages, translations)):
            progress((i + 1) * 0.1 + 0.5, desc=f"Generating {lang} audio...")
            try:
                audio_path = readtranslation(translation, audiofile, lang)
                audio_paths.append(audio_path)
            except Exception as e:
                print(f"Error generating audio for {lang}: {str(e)}")
                audio_paths.append(None)
        
        progress(1.0, desc="Process complete!")
        return audio_paths + translations
    except Exception as e:
        raise gr.Error(f"An error occurred: {str(e)}")
    finally:
        cleanup_memory()

with gr.Blocks() as demo:
    gr.Markdown("## Record yourself in any language and immediately receive voice translations.")
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"],
                                   type="filepath",
                                   show_download_button=True,
                                   waveform_options=gr.WaveformOptions(
                                       waveform_color="#01C6FF",
                                       waveform_progress_color="#0066B4",
                                       skip_length=2,
                                       show_controls=False,
                                   ))
            with gr.Row():
                submit = gr.Button("Submit", variant="primary")
                btn = gr.ClearButton(audio_input, "Clear")

    with gr.Row():
        with gr.Group():
            es_output = gr.Audio(label="Spanish", interactive=False)
            es_text = gr.Markdown()
        with gr.Group():
            fr_output = gr.Audio(label="French", interactive=False)
            fr_text = gr.Markdown()
        with gr.Group():
            hi_output = gr.Audio(label="Hindi", interactive=False)
            hi_text = gr.Markdown()

    output_components = [es_output, fr_output, hi_output,
                         es_text, fr_text, hi_text]
    submit.click(fn=voice_to_voice, inputs=audio_input, outputs=output_components, show_progress=True)

def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory cleaned up")

if __name__ == "__main__":
    demo.launch()
    cleanup_memory()