import openai
import requests
from gtts import gTTS
from pygame import mixer
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import en_core_web_sm
from pynput.keyboard import Key, Listener
import os
import time
import pygame
import speech_recognition as sr
import pyaudio
import wave
import time
from contextlib import contextmanager
import sys
import msvcrt
from transformers import pipeline
import torch
from string import punctuation


# Stuff for custom voice
from Audio_Generation.Generation_Scripts.generation import str_enc, Audio_Obj





def test_audio():
    model_fpath = 'Audio_Generation/Generation_Scripts/saved_models/default/encoder.pt'
    synth_path = 'Audio_Generation/Generation_Scripts/saved_models/default/synthesizer.pt'
    vocode_path = 'Audio_Generation/Generation_Scripts/saved_models/default/vocoder.pt'

    # Create a new object
    obj = Audio_Obj(model_fpath, synth_path, vocode_path)

    # Load in a file
    p = "Audio_Generation/Generation_Scripts/data/albedo/"
    obj.load_from_browser("1.5.mp3", p)
    obj.load_from_browser("2.5.mp3", p)
    obj.load_from_browser("3.5.mp3", p)
    obj.load_from_browser("4.5.mp3", p)
    obj.load_from_browser("5.5.mp3", p)
    obj.load_from_browser("6.5.mp3", p)
    obj.load_from_browser("7.5.mp3", p)
    obj.load_from_browser("8.5.mp3", p)
    
    # obj.load_from_browser("1.mp3", "data/shylily")
    # obj.load_from_browser("2.mp3", "data/shylily")
    # obj.load_from_browser("3.mp3", "data/shylily")
    # obj.load_from_browser("4.mp3", "data/shylily")
    # obj.load_from_browser("5.mp3", "data/shylily")
    # obj.load_from_browser("6.mp3", "data/shylily")
    # obj.load_from_browser("7.mp3", "data/shylily")
    # obj.load_from_browser("8.mp3", "data/shylily")
    # obj.load_from_browser("9.mp3", "data/shylily")
    # obj.load_from_browser("10.mp3", "data/shylily")
    # obj.load_from_browser("11.mp3", "data/shylily")
    # obj.load_from_browser("12.mp3", "data/shylily")
    # obj.load_from_browser("13.mp3", "data/shylily")
    # obj.load_from_browser("14.mp3", "data/shylily")
    # obj.load_from_browser("15.mp3", "data/shylily")
    # obj.load_from_browser("16.mp3", "data/shylily")
    # obj.load_from_browser("17.mp3", "data/shylily")
    # obj.load_from_browser("18.mp3", "data/shylily")
    # obj.load_from_browser("19.mp3", "data/shylily")
    # obj.load_from_browser("20.mp3", "data/shylily")
    # obj.load_from_browser("21.mp3", "data/shylily")
    # obj.load_from_browser("22.mp3", "data/shylily")
    # obj.load_from_browser("23.mp3", "data/shylily")
    # obj.load_from_browser("24.mp3", "data/shylily")


    while True:
        # Get the text
        print("Prompt: ", end="")
        text = input()

        if text == "":
            break
        
        # Any punctuation is replaced with a newline
        text = text.replace(".", "\n").replace("?", "\n").replace("!", "\n")

        # Create the audio
        obj.synthesize(text)
        obj.vocode()











# Stop annoying things from outputting
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


# Puncuation tokenizer
tokenizer = RegexpTokenizer(r'\w+')

# VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Audio recognizer
r = sr.Recognizer()

# Get the sentiment of text
def get_sent(text):
    sents = sia.polarity_scores(text)
    
    if sents["neg"] > 0.5:
        return "angry"
    elif sents["pos"] > 0.5:
        return "happy"
    else:
        return ""


# Summary function
def summarize(text, per):
    nlp = spacy.load('en_core_web_sm')
    doc= nlp(text)
    tokens=[token.text for token in doc]
    word_frequencies={}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_frequency
    sentence_tokens= [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():                            
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    select_length=int(len(sentence_tokens)*per)
    summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
    final_summary=[word.text for word in summary]
    summary=''.join(final_summary)
    return summary


# Get the summary of the text
def get_summ(text):
    # Get the summary
    summary = summarize(text, 0.5)
    
    # Remove stopwords and puncuation from the summary
    filtered = [word for word in tokenizer.tokenize(summary) if word not in stopwords.words('english')]
    
    return " ".join(filtered)

# Build a prompt for the image
def build_img_prompt(text):
    # Get the summary and sentiment
    sent = get_sent(text)
    summary = get_summ(text)
    
    # Create the image prompt
    settings = "1girl, very wide shot, simple background, solo focus, feamle focus, looking at viewer, ratio:16:9, realistic, detailed"
    characteristics = "waifu, female, brown hair, blue eyes, sidelocks, slight blush, fox ears"
    # sent = "furious"
    # summary = "'I hope get know better' to viewer"
    prompt = f"{settings} {characteristics} {','+sent if len(sent)!=0 else ''}, {summary}"
    return prompt

# Get the audio input from the user
def get_audio_input():
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second
    seconds = 10
    filename = "tmp.wav"
    global enter_pressed

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks until enter is hit
    time.sleep(0.5)
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
        if enter_pressed:
            enter_pressed = False
            break

            # Stop and close the stream 
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    # Open the wav file and read in the data
    # Get the audio data
    audio = sr.AudioFile("tmp.wav")
    with audio as source:
        audio = r.record(source)
        
    # Get the text from the audio
    with suppress_stdout():
        try:
            text = r.recognize_google(audio)
        except sr.UnknownValueError:
            return ""
    
    return text

# Get the response from GPT-3
def get_response_gpt(text):
    # Open AI Key
    openai.api_key = "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"
    
    output = openai.Completion.create(
      model="text-davinci-003",
      prompt=text,
      max_tokens=50,
      temperature=0.7
    )
    
    # Only allow the response to be one line
    resp = output["choices"][0]["text"].lstrip().split("\n")[0]
    
    return resp

# Load in the other model
other_model = pipeline('text-generation',model="Finetuning/outputs/r/",
                      tokenizer='EleutherAI/gpt-neo-1.3B',max_new_tokens=50,
                      torch_dtype=torch.float16,framework="pt",
                      device=torch.device("cuda:0"))

# Get a response from the other model
def get_response_other(text):
    # How many newlines are there?
    num = text.count("\n")

    # Get the model output. at the correct position
    output = other_model(text)[0]['generated_text'].split("\n")
    output_new = output[num].strip()

    # Make sure the output is not blank
    tmp = 1
    #output_new = output_new.replace("You:", "").replace("Person:", "")
    while output_new == "":
        output_new = output[num+tmp].strip()
        tmp += 1
        
    # If the model is generating newlines after its text,
    # it may want to say more
    cur_out = output_new
    more_max = 0 # Max limit on how much more to add
    more_added = 0 # Current extra added
    while more_added < more_max:
        try:
            if output[num+tmp].strip() == "":
                break # Break is a \n\n is reached. Keep going if only \n
            out_new = output[num+tmp].strip()
            if out_new not in punctuation:
                out_new += "."
            cur_out += f" {out_new}"
            more_added += 1
            tmp += 1

            # If a question make was the last letter,
            # stop adding more lines
            if cur_out[-1] == "?":
                break
        except IndexError:
            break

    return cur_out

# Get the image generation model
pipe = StableDiffusionPipeline.from_pretrained(
    'hakurei/waifu-diffusion',
    torch_dtype=torch.float16,
    cache_dir="D:/python-libs/hugging-face-cache",
).to('cuda')
# Remove filter
pipe.safety_checker = lambda images, clip_input: (images, False)


# Load the custom audio models
audioObj = None
def load_custom_audio():
    global audioObj

    model_fpath = 'Audio_Generation/Generation_Scripts/saved_models/default/encoder.pt'
    synth_path = 'Audio_Generation/Generation_Scripts/saved_models/default/synthesizer.pt'
    vocode_path = 'Audio_Generation/Generation_Scripts/saved_models/default/vocoder.pt'

    # Create a new object
    audioObj = Audio_Obj(model_fpath, synth_path, vocode_path)

    # Load in a file
    p = "Audio_Generation/Generation_Scripts/data/albedo/"
    audioObj.load_from_browser("1.5.5.mp3", p)
    audioObj.load_from_browser("2.5.mp3", p)
    audioObj.load_from_browser("3.5.mp3", p)
    audioObj.load_from_browser("4.5.mp3", p)
    audioObj.load_from_browser("5.5.mp3", p)
    audioObj.load_from_browser("6.5.mp3", p)
    audioObj.load_from_browser("7.5.mp3", p)
    audioObj.load_from_browser("8.5.mp3", p)

    # Create the audio
    print("Testing custom audio...")
    audioObj.synthesize("Hello there")
    audioObj.vocode(play_audio=False)
    print("Testing complete")


# Create the audio clip
def create_audio(text, custom_audio):
    global audioObj
    if custom_audio:
        audioObj.synthesize(text)
        audioObj.vocode(play_audio=False)
    else:
        myobj = gTTS(text=text, lang='en', slow=False)
        myobj.save("tmp.mp3")


def main():
    # Use custom audio or not
    custom_audio = True

    # Use custom model or GPT3
    custom_model = True

    # Load in the custom audio
    if custom_audio:
        load_custom_audio()

    global space_pressed
    # The prompt is initially a basic prompt telling GPT-3 who it is
    prompt = "You are my female waifu girlfriend who loves me\n\n\n\n"\
        "Me: Hi\nYou: Hello\n\n"\
        "Me: How are you?\nYou: Good. How are you?\n\n"\
        "Me: I'm good.\nYou: Nice to meet you.\n\n"
    
    mixer.init()
    mixer.music.unload()
    while True:
        # Wait for person to press space
        print("Press space to talk to my waifu")
        while space_pressed == False:
            pass
        space_pressed = True
        print("Press enter when done speaking")
        
        # Get the audio input
        text_prompt = get_audio_input()
        
        if len(text_prompt) < 3:
            print("No audio detected. Try typing instead")

            sys.stdout.flush()
            # Try to flush the buffer
            while msvcrt.kbhit():
                msvcrt.getch()
            time.sleep(1)
            
            # Wait for a prompt to be entered
            print("Prompt: ", end="")
            text_prompt = input()
        
        # Add the text to the current prompt
        prompt += f"Me: {text_prompt}\n"
        
        # Get the text from the model
        if custom_model == True:
            ret_text = get_response_other(prompt)
        else:
            ret_text = get_response_gpt(prompt)

        # Sometimes a stupid output will be placed at the
        # beginning like [Random name]: [words].
        # let's remove these
        ret_text = ret_text.split(":")[-1]

        print(ret_text)
        
        # Create audio for the returned text
        if len(ret_text) > 3:
            # Create the audio clip
            pygame.mixer.stop()
            mixer.music.unload()
            create_audio(ret_text, custom_audio)
            
            # Play the audio
            try:
                mixer.music.load('tmp.mp3')
                mixer.music.play()
            except pygame.error:
                s = mixer.Sound('tmp.mp3')
                s.play()
            
            # Get the image prompt
            img_prompt = build_img_prompt(ret_text)
            
            # Get the image
            with suppress_stdout():
                with autocast("cuda"):
                    image = pipe(img_prompt, guidance_scale=10)["images"][0]

            # clear_output(wait=True)
        
            # Show the image
            fig, ax = plt.subplots()
            fig.subplots_adjust(0,0,1,1)
            ax.set_axis_off()
            ax.imshow(image)
            plt.show()
            del image
        
        # Add the new text to the prompt
        prompt += f"You: {ret_text}\n"
    





# Is enter or space pressed?
enter_pressed = False
space_pressed = False

def on_press(key):
    global enter_pressed
    global space_pressed
    if key == Key.enter:
        enter_pressed = True
    elif key == Key.space:
        space_pressed = True
def on_release(key):
    global enter_pressed
    global space_pressed
    enter_pressed = False
    space_pressed = False



if __name__=="__main__":
    # Collect events until released
    listener = Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()

    # test_audio()
    main()