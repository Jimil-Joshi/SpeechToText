import logging
import requests as re
import torchaudio
from transformers import SpeechT5Processor, SpeechT5ForSpeechToText
import os
import shutil
import requests
from pydub import AudioSegment
import detect_silence
from datetime import datetime

current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
str_current_datetime = str(current_datetime)

logging.getLogger('tensorflow').setLevel(logging.ERROR)

processor = SpeechT5Processor.from_pretrained("./speecht5_asr")
model = SpeechT5ForSpeechToText.from_pretrained("./speecht5_asr")


def load_audio(audio_file):
    """Load the audio file & convert to 16,000 sampling rate"""
    # load our wav file
    audio_path = re.get(audio_file)
    with open("sample.wav", "wb") as file:
        file.write(audio_path.content)
    file.close()
    speech, sr = torchaudio.load("sample.wav")
    resampler = torchaudio.transforms.Resample(sr, 16000)
    sp = resampler(speech)
    final_speech = sp.squeeze(0)
    # file.close()
    return final_speech


def load_audio_local(audio_file):
    """Load the audio file & convert to 16,000 sampling rate"""
    speech, sr = torchaudio.load(audio_file)
    resampler = torchaudio.transforms.Resample(sr, 16000)
    sp = resampler(speech)
    final_speech = sp.squeeze(0)
    # file.close()
    return final_speech, sr


def download_audio_from_url(url, destination_folder):
    """Downloads audio from a URL and saves it to the specified folder."""
    response = requests.get(url)
    if response.status_code == 200:
        file_path = os.path.join(destination_folder, str_current_datetime + "downloaded_audio.wav")
        with open(file_path, 'wb') as audio_file:
            audio_file.write(response.content)
        return file_path
    else:
        raise Exception("Failed to download audio from the URL.")


def transcribe_audio(audio_file):
    try:
        # Attempt to load audio using speech,sr = load_audio(audio_file)
        speech, sr = load_audio(audio_file)
    except:
        # If an error occurs, try loading audio using speech = load_audio(audio_file)
        speech = load_audio(audio_file)
        sr = 16000  # Set the default sampling rate to 16000 if it's not available in load_audio()
    # speech,sr = load_audio(audio_file)
    input_features = processor(audio=speech,
                               sampling_rate=16000,
                               return_tensors="pt"
                               )
    # max_new_tokens=450
    generated_ids = model.generate(inputs=input_features["input_values"],
                                   attention_mask=input_features["attention_mask"],
                                   max_new_tokens=70
                                   )
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    # print(type(transcription))
    # print(transcription)
    words = transcription[0].split(",")

    # Remove duplicates while preserving order
    unique_words = []
    for word in words:
        if word not in unique_words:
            unique_words.append(word)

    # Join the unique words back into a string
    transcription_result = ' '.join(unique_words)

    # print(transcription)

    return transcription_result


def transcribe_audio_V2(audio_file):
    """
    This function works with penalty and temperateure and max tokens is 450
    This is relatively slower than the proevious version but more accurate.
    :param audio_file: str
    :return: transcription , str
    """
    speech, sr = load_audio_local(audio_file)
    input_features = processor(audio=speech,
                               sampling_rate=16000,
                               return_tensors="pt"
                               )

    generated_ids = model.generate(inputs=input_features["input_values"],
                                   attention_mask=input_features["attention_mask"],
                                   max_new_tokens=450,
                                   repetition_penalty=1.0,
                                   temperature=0.9,
                                   )

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return transcription


def transcribe_audio_largefile(url):
    """Downloads audio from a URL, splits it into chunks, and transcribes them."""
    # Create a temporary folder to store the audio chunks
    folder_name = "audio-chunks"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Download the audio file from the URL
    downloaded_audio_path = download_audio_from_url(url, folder_name)

    # Load the downloaded audio
    sound = AudioSegment.from_file(downloaded_audio_path)

    # Split audio sound where silence is 500 milliseconds or more and get chunks
    chunks = detect_silence.split_on_silence(
        sound,
        min_silence_len=500,
        silence_thresh=sound.dBFS - 15,
        keep_silence=500,
    )

    whole_text = ""

    # Process each chunk
    for i, audio_chunk in enumerate(chunks, start=1):
        # Export audio chunk and save it in the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, str_current_datetime + f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")

        # Recognize the chunk (you'll need to replace this with your actual transcription logic)
        try:
            text = transcribe_audio_V2(chunk_filename)
            text_str = " ".join(text)
        except Exception as e:
            print("Error:", str(e))
            text_str = f"Error transcribing chunk {i}."

        text_str = f"{text_str.capitalize()}. "
        # print(chunk_filename, ":", text_str)
        whole_text += text_str

        # Remove the processed chunk
        os.remove(chunk_filename)

        # Remove the temporary folder
    shutil.rmtree(folder_name)

    # Return the text for all chunks detected
    return whole_text.strip()


if __name__ == '__main__':
    # audio_file = "http://172.21.102.211/aidata/tele_health/inputs/2.wav"
    audio_file = "http://172.21.102.211/aidata/old_dxc/speech/transcribe_audio/med_history.wav"

    text = transcribe_audio(audio_file)
    print(text)