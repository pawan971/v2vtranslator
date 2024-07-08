import gradio as gr
import whisper
from translate import Translator
from TTS.api import TTS
import uuid
import os
from pathlib import Path


model = whisper.load_model("base")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
os.environ["COQUI_TOS_AGREED"] = "1"

def v2vtranslate(audiofile):

    print("Starting transcription...")
    transcription_result = transcribeaudio(audiofile)

    if transcription_result.status == model.transcribe.error:
      raise gr.Error(transcription_result.error)
    else:
      text = transcription_result.text
      print(f"Transcribed Text: {text}")

    print("Starting translation...")
    es_translation,fr_translation,hi_translation,ja_translation = translatetext(text)
    print(f"Translations:\nSpanish: {es_translation}\nFrench: {fr_translation}\nHindi: {hi_translation}\nJapanese: {ja_translation}")

    print("Generating TTS audio files(Outside Function)...")
    es_translation_path = readtranslation(es_translation,audiofile)
    fr_translation_path = readtranslation(fr_translation,audiofile)
    hi_translation_path = readtranslation(hi_translation,audiofile)
    ja_translation_path = readtranslation(ja_translation,audiofile)
    print(f"Generated audio paths:\nSpanish: {es_translation_path}\nFrench: {fr_translation_path}\nHindi: {hi_translation_path}\nJapanese: {ja_translation_path}")



    es_path = Path(es_translation_path)
    fr_path = Path(fr_translation_path)
    hi_path = Path(hi_translation_path)
    ja_path = Path(ja_translation_path)



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
    print(f"Detected language: {max(probs, key=probs.get)}")

    return tresult

def translatetext(text):

    translator_spanish = Translator(from_lang="en",to_lang="es")
    es_text = translator_spanish.translate(text)

    translator_french = Translator(from_lang="en",to_lang="fr")
    fr_text = translator_french.translate(text)

    translator_hindi = Translator(from_lang="en",to_lang="hi")
    hi_text = translator_hindi.translate(text)

    translator_japanese = Translator(from_lang="en",to_lang="ja")
    ja_text = translator_japanese.translate(text)
    print(f"Japanese Translation(Inside Function): {ja_text}")

    return es_text,fr_text,hi_text,ja_text


def readtranslation(text,audiofile):

    print(f"Generating TTS for text(Inside Function): {text}")
    output_path = f"{uuid.uuid4()}.wav"
    tts.tts_to_file(text=text,
                file_path=output_path,
                speaker_wav=audiofile,
                language="en")
    print(f"Generated audio file at: {output_path}")
    return output_path


audio_input = gr.Audio(
    sources=['microphone'],
    type="filepath"
    )
demo = gr.Interface(
    fn=v2vtranslate,
    inputs=audio_input,
    outputs=[gr.Audio(label="Spanish"),gr.Audio(label="French"),gr.Audio(label="Hindi"),gr.Audio(label="Japanese")]
)

if __name__ == "__main__":
    demo.launch()