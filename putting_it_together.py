import openai
from gtts import gTTS
import pygame
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
from pynput.keyboard import Key, Listener
import os
import time
import speech_recognition as sr
import pyaudio
import wave
from contextlib import contextmanager
import sys
import msvcrt
from transformers import pipeline
from string import punctuation
from keybert import KeyBERT


# Stuff for custom voice
from Audio_Generation.Generation_Scripts.generation import Audio_Obj





def test_audio(audio_model_path, audio_data_path):
    model_fpath = f'{audio_model_path}{os.sep}encoder.pt'
    synth_path = f'{audio_model_path}{os.sep}synthesizer.pt'
    vocode_path = f'{audio_model_path}{os.sep}vocoder.pt'

    # Create a new object
    obj = Audio_Obj(model_fpath, synth_path, vocode_path)

    # Load in a file
    obj.load_from_browser("1.5.5.mp3", audio_data_path)
    obj.load_from_browser("2.5.mp3", audio_data_path)
    obj.load_from_browser("3.5.mp3", audio_data_path)
    obj.load_from_browser("4.5.mp3", audio_data_path)
    obj.load_from_browser("5.5.mp3", audio_data_path)
    obj.load_from_browser("6.5.mp3", audio_data_path)
    obj.load_from_browser("7.5.mp3", audio_data_path)
    obj.load_from_browser("8.5.mp3", audio_data_path)
    
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
def summarize(text):
    global summ_model

    # Get the keywords
    keywords = summ_model.extract_keywords(text)

    # Get keywords above a threshold
    words = ", ".join([word[0] for word in keywords])

    return words


# Get the summary of the text
def get_summ(text):
    # Get the summary
    summary = summarize(text)
    
    # Remove stopwords and puncuation from the summary
    filtered = [word for word in tokenizer.tokenize(summary) if word not in stopwords.words('english')]
    
    return " ".join(filtered)

# Build a prompt for the image
def build_img_prompt(text, settings, characteristics):
    # Get the summary and sentiment
    sent = get_sent(text)
    summary = get_summ(text)
    
    # Create the image prompt
    # settings = "1girl, very wide shot, simple background, solo focus, female focus, looking at viewer, ratio:16:9, detailed"
    # characteristics = "waifu, female, brown hair, blue eyes, sidelocks, slight blush, fox ears"
    # sent = "furious"
    # summary = "'I hope get know better' to viewer"
    prompt = f"{settings}, {characteristics}, {sent+', ' if sent != '' else ''}'{summary}' to viewer"
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
def get_response_gpt(text, GPT_key):
    # Open AI Key
    openai.api_key = GPT_key
    
    output = openai.Completion.create(
      model="text-davinci-003",
      prompt=text,
      max_tokens=50,
      temperature=0.7
    )

    openai.api_key = None
    
    # Only allow the response to be one line
    resp = output["choices"][0]["text"].lstrip().split("\n")[0]
    
    return resp

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


# Load the custom audio models
def load_custom_audio(audio_model_path, audio_data_path):
    global audioObj
    model_fpath = f'{audio_model_path}{os.sep}encoder.pt'
    synth_path = f'{audio_model_path}{os.sep}synthesizer.pt'
    vocode_path = f'{audio_model_path}{os.sep}vocoder.pt'

    # Create a new object
    audioObj = Audio_Obj(model_fpath, synth_path, vocode_path)

    # Load in a file
    audioObj.load_from_browser("1.5.mp3", audio_data_path)
    audioObj.load_from_browser("2.5.mp3", audio_data_path)
    audioObj.load_from_browser("3.5.mp3", audio_data_path)
    audioObj.load_from_browser("4.5.mp3", audio_data_path)
    audioObj.load_from_browser("5.5.mp3", audio_data_path)
    audioObj.load_from_browser("6.5.mp3", audio_data_path)
    audioObj.load_from_browser("7.5.mp3", audio_data_path)
    audioObj.load_from_browser("8.5.mp3", audio_data_path)

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


def main(custom_audio, custom_model, img_settings, img_characteristics, GPT_key):
    # The prompt is initially a basic prompt telling GPT-3 who it is
    prompt = "You are my female waifu girlfriend who loves me\n\n\n\n"\
        "Me: Hi\nYou: Hello\n\n"\
        "Me: How are you?\nYou: Good. How are you?\n\n"\
        "Me: I'm good.\nYou: Nice to meet you.\n\n"
    
    global space_pressed
    global pipe

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
            ret_text = get_response_gpt(prompt, GPT_key)

        # Sometimes a stupid output will be placed at the
        # beginning like [Random name]: [words].
        # let's remove these
        ret_text = ret_text.split(":")[-1]

        print(ret_text)
        
        # Create audio and the image for the returned text
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
            img_prompt = build_img_prompt(ret_text, img_settings, img_characteristics)
            
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







# Params:
#   audio_model_path - Path to the custom audio model
#   audio_data_path - Path to the custom audio data
#   custom_model_path - Path to the custom model to load in
def setup(audio_model_path=None, audio_data_path=None, custom_model_path=None):
    # Puncuation tokenizer
    global tokenizer
    tokenizer = RegexpTokenizer(r'\w+')

    # VADER sentiment analyzer
    global sia
    sia = SentimentIntensityAnalyzer()

    # Audio recognizer
    global r
    r = sr.Recognizer()



    # Is enter or space pressed? Currently
    # this is False
    global enter_pressed
    global space_pressed
    enter_pressed = False
    space_pressed = False

    # Used to collect keyboard events
    listener = Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()



    # Initialize the summary model
    global summ_model
    summ_model = KeyBERT()


    

    # Image generation model
    print("Initializing image model...")
    global pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        'hakurei/waifu-diffusion',
        torch_dtype=torch.float16,
        cache_dir="D:/python-libs/hugging-face-cache",
    ).to('cuda')
    # Remove filter
    pipe.safety_checker = lambda images, clip_input: (images, False)
    print("Image model initialized!")



    # Get the image generation model if
    # the GPT model is notused
    assert custom_model_path != None, "Custom model path cannot be none if using a custom model"
    print("Initializing custom text model")
    global other_model
    other_model = pipeline('text-generation',model=custom_model_path,
                    tokenizer='EleutherAI/gpt-neo-1.3B',max_new_tokens=50,
                    torch_dtype=torch.float16,framework="pt",
                    device=torch.device("cuda:0"))
    print("Custom text model initialized!")
    # Otherwise, use the GPT model

    

    # Audio object is initially None, but
    # will be replace later
    global audioObj
    audioObj = None

    # Load in the custom audio
    print("Initializing custom audio model")
    assert audio_data_path != None, "Audio data path needs to be specified if using custom audio"
    assert audio_model_path != None, "Audio model path needs to be specified is using custom audio"
    load_custom_audio(audio_model_path, audio_data_path)
    print("Custom audio model initialized!")

    # Initialize the audio mixer so audio can be played
    mixer.init()
    mixer.music.unload()







if __name__=="__main__":
    # Use custom audio or not
    custom_audio = True

    # Use custom model or GPT3
    custom_model = True

    # Path to the custom audio model
    audio_model_path = "Audio_Generation/Generation_Scripts/saved_models/default"

    # Path to the custom audio data
    audio_data_path = "Audio_Generation/Generation_Scripts/data/albedo"

    # Path to the custom model to load in
    custom_model_path = "Finetuning/outputs/r/"

    # Settings and characteristics for the output image
    img_settings = "1girl, very wide shot, simple background, solo focus, female focus, looking at viewer, ratio:16:9, detailed"
    img_characteristics = "waifu, female, brown hair, blue eyes, sidelocks, slight blush, fox ears"
    
    # Setup the interface
    setup(audio_model_path, audio_data_path, custom_model_path)
    #test_audio(audio_model_path, audio_data_path)

    # Run the interface
    GPT_key = "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"
    main(custom_audio, custom_model, img_settings, img_characteristics, GPT_key)