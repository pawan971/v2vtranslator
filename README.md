# Multilingual Voice-to-Voice Translation App 🎙️🔊

## Overview

This repository hosts a open source Gradio-based application that translates your voice into multiple languages. The app leverages state-of-the-art models for transcription, language detection, translation, and text-to-speech synthesis to provide an end-to-end solution for real-time multilingual voice translation.

## Features

- **Transcription**: Converts spoken language into written text.
- **Automatic Language Detection**: Detects the language of the spoken input automatically.
- **Translation**: Translates the transcribed text into a selected target language.
- **Text-to-Speech**: Converts the translated text back into speech, mimicking the original speaker's voice as closely as possible.
- **Language Selection**: Supports 17 languages for translation and speech synthesis.
- **User Agreement**: Includes an option to agree to the COQUI terms and conditions before using the service.

## Installation/Usage

### Method 1 (HTTPS)
 
App currently hosted on HuggingFace Spaces. Use the link below to access: 

[v2vtranslator - HugginFace Spaces](https://huggingface.co/spaces/DhakkiTikki/v2vtranslator)

### Method 2 (local)

1. Clone the repository:

    ```sh
    git clone https://github.com/pawan971/v2vtranslator
    cd v2vtranslator
    ```

2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

3. Run the application:

    ```sh
    python app.py
    ```

## Models Used

- **Whisper**: Used for audio transcription and automatic language detection.
- **translate**: Used for text translation between languages.
- **XTTS-v2**: Used for text-to-speech synthesis to generate audio from translated text in your voice.

## Open Source

This project is open source and contributions are welcome! Feel free to open issues, submit pull requests, or fork the repository to add your own features.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

Special thanks to the developers of Whisper, translate, and XTTS-v2 for providing the foundational models used in this application.

---
