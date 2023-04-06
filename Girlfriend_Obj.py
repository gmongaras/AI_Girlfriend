"""
This class is the main class pretty much just holding
a bunch of functions needed to make the program work.
"""



# NLTK downloads
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')


import openai
from gtts import gTTS
from gtts.tts import gTTSError
from pygame import mixer
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from nltk.sentiment import SentimentIntensityAnalyzer
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import os
import time
import speech_recognition as sr
import pyaudio
import wave
from contextlib import contextmanager
import sys
from transformers import pipeline
from string import punctuation
from keybert import KeyBERT
import json
from vosk import KaldiRecognizer
from vosk import Model as vosk_Model
from pydub import AudioSegment
import numpy as np
import asyncio
from PIL import Image
from Img_Mover.Img_Mover import Img_Mover
import pygame
import gradio as gr
import threading
from copy import deepcopy
import math






class Girlfriend_Obj:
    # Params:
    #   initial_summary - String to initialize the summary to (tells
    #                    the model who it is)
    #   initial_prompt - String to initialize the prompt to (tells
    #                    the model how to respond)
    #   load_custom_audio - True to load custom audio. False otherwise
    #   audio_model_path - Path to the custom audio model
    #   audio_data_path - Path to the custom audio data
    #   custom_model_path - Path to the custom model to load in
    #   saved_memory - (Optionl) path to the json file with the
    #                   saved memory to load in
    def __init__(self, initial_summary="", initial_prompt="", load_custom_audio=False, audio_model_path=None, audio_data_path=None, custom_model_path=None, saved_memory=None):
        # Save the initial configuration in case
        # the user resets the memory
        self.initial_summary = initial_summary
        self.initial_prompt = initial_prompt
        
        
        """Class Globals"""
        # Used to stop the blinking loop if needed
        self.stop_animating = False
        # Used to tell the rest of the code if
        # mouth movement is being generated or not
        self.generating_mouth_movement = False
        # Used to store the last image and text generated
        # in case of errors
        self.last_image = np.zeros((50, 50, 3))
        self.last_text = "Error"
        # Should movement be added to the image or not?
        self.add_movement = True
        # Should a new image be forcefully generated?
        self.force_gen = False
        # Holds a thread moving the mouth if any
        self.m_thread = None
        # Holds the blink thread if any
        self.b_thread = None



        """Small models being used"""

        # Audio recognizer
        self.recognizer = sr.Recognizer()

        # Puncuation tokenizer
        self.tokenizer = RegexpTokenizer(r'\w+')

        # VADER sentiment analyzer
        self.sent_model = SentimentIntensityAnalyzer()

        # Audio recognizer
        self.audio_recognizer = sr.Recognizer()

        # Initialize the summary model
        self.summ_model = KeyBERT()

        # Vosk SST model
        model_path = "vosk_models/vosk-model-small-en-us-0.15"
        self.vosk_model = vosk_Model(model_path)


        



        """Thick models being used"""

        # Image generation model
        print("Initializing image model...")
        self.imgGen = StableDiffusionPipeline.from_pretrained(
            'hakurei/waifu-diffusion',
            torch_dtype=torch.float16
        ).to('cuda')
        # Remove filter
        self.imgGen.safety_checker = lambda images, clip_input: (images, False)
        print("Image model initialized!")

        # Get the image generation model if
        # the GPT model is not used
        # https://huggingface.co/gmongaras/gpt-anime-sub-1.3B/
        # https://huggingface.co/EleutherAI/gpt-neo-1.3B/
        # Max len is 2048
        print("Initializing custom text model")
        self.other_text_model = pipeline('text-generation',model="gmongaras/gpt-anime-sub-1.3B",
                      tokenizer="EleutherAI/gpt-neo-1.3B",
                      max_new_tokens=50,
                      torch_dtype=torch.float16,framework="pt",
                      device=torch.device("cuda:0"),
                      pad_token_id=50256)
        print("Custom text model initialized!")
        # Otherwise, use the GPT model

        # Load in the large summarizer model
        # https://huggingface.co/pszemraj/led-large-book-summary
        print("Initializing summarizer...")
        self.summarizer = pipeline(
            "summarization",
            'pszemraj/led-large-book-summary',
            device=0 if torch.cuda.is_available() else -1,
            framework="pt",
            torch_dtype=torch.float16,
        )
        print("Summarizer initialized!")







        """
        The summary has three parts:
        1. Summary of the entire past
            - This is the summary of the entire past that
              cannot be kept in memory and is a set size
        2. Multiple past output sequences
            - This part is just a bunch of past outputs
              that are larger than the summary of the past
            - This part is used for two things:
                1. Having a more detailed past
                2. Few (well actually a lot) of
                   example to show the model how to respond
            - This part is broken into several subsections which
              are just a way to split this into blocks that
              can be fed into the summary. In a way, this
              is a queue with the oldest history being fed
              into the summary as needed to keep the output
              from becoming too large
        3. Current output
            - This is similar to 2, but is not currently
              in the queue. It won't enter the queue until
              it reaches a certain size
        """
        # Sizes of the three parts
        self.summ_size_max = 256 # Used for 1
        self.block_size = 150 # Used for 2 and 3
        self.num_blocks = 4 # Used for 2

        # the three parts
        self.past_summ = initial_summary #  1
        self.past_output = ["" for i in range(self.num_blocks)] # 2
        self.cur_prompt = initial_prompt # 3

        # Dictionary used to save the model state
        self.disc_json = dict(
            past_summ=self.past_summ,
            past_output=self.past_output,
            cur_prompt=self.cur_prompt,
        )

        # Load in the memory if there is any
        if saved_memory is not None:
            self.load_memory(saved_memory)

        

        # Audio object is initially None, but
        # may be replace later if using custom audio
        self.audioObj = None

        # Load in the custom audio
        if load_custom_audio == True:
            assert custom_model_path != None, "Custom model path cannot be none if using a custom model"
            print("Initializing custom audio model")
            assert audio_data_path != None, "Audio data path needs to be specified if using custom audio"
            assert audio_model_path != None, "Audio model path needs to be specified is using custom audio"
            load_custom_audio(audio_model_path, audio_data_path)
            print("Custom audio model initialized!")
        else:
            print("Not loading custom audio model")

        # Initialize the audio mixer so audio can be played
        try:
            mixer.init()
            mixer.music.unload()
        except:
            pass




        # Used to work with image movement
        # Load in the default image
        img = Image.open("default_img.png")
        self.last_image = img
        # Create the class to add image movement
        print("Initializing custom image movement module")
        self.img_anim = Img_Mover(torch.device("cuda:0"), 0.5, automatic_EMA=True)
        # Load in the new image
        self.img_anim.load_new_image(img=img)
        # Default pose for the image
        self.img_anim.change_pose()
        print("Image movement module initialized!")





        # Dummy variable where the thread
        # that animates the picture can be accessed from
        self.anim_thread = None









    # Load in memory from a saved file
    def load_memory(self, filename):
        # Load in the dictionary
        self.disc_json = json.load(open(filename, "r"))
        assert "past_summ" in self.disc_json, "Loaded file must have past_summ key"
        assert "past_output" in self.disc_json, "Loaded file must have past_output key"
        assert "cur_prompt" in self.disc_json, "Loaded file must have cur_prompt key"

        # Extract the data
        self.past_summ = self.disc_json["past_summ"]
        self.past_output = self.disc_json["past_output"]
        self.cur_prompt = self.disc_json["cur_prompt"]

    
    # Load in a memory file with error checking
    def load_mem(self, filename):
        try:
            self.load_memory(filename)
            return "Success!"
        except:
            return "Fail! File does not exist or is in incorrect format"



    # Stop annoying things from outputting
    @contextmanager
    def suppress_stdout(self):
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:  
                yield
            finally:
                sys.stdout = old_stdout

    # Get the sentiment of text and return it as a string (happy or angry)
    def get_sent(self, text):
        sents = self.sent_model.polarity_scores(text)
        
        if sents["neg"] > 0.5:
            return "angry"
        elif sents["pos"] > 0.5:
            return "happy"
        else:
            return ""


    # Summary function for a single line using a small model
    def summarize_single(self, text):
        # Get the keywords
        keywords = self.summ_model.extract_keywords(text)

        # Get keywords above a threshold
        words = ", ".join([word[0] for word in keywords])

        return words


    # Get the summary of the text using the large model
    def get_summ(self, text):
        # Get the summary
        summary = self.summarize_single(text)
        
        # Remove stopwords and puncuation from the summary
        filtered = [word for word in self.tokenizer.tokenize(summary) if word not in stopwords.words('english')]
        
        return " ".join(filtered)

    # Given specific parts of the imag prompt, build out the image prompt
    # and return it
    def build_img_prompt(self, text, settings, characteristics):
        # Get the summary and sentiment
        sent = self.get_sent(text)
        summary = self.get_summ(text)
        
        # Create the image prompt
        # settings = "1girl, very wide shot, simple background, solo focus, female focus, looking at viewer, ratio:16:9, detailed"
        # characteristics = "waifu, female, brown hair, blue eyes, sidelocks, slight blush, fox ears"
        # sent = "furious"
        # summary = "'I hope get know better' to viewer"
        img_prompt = f"{settings} {characteristics} {','+sent if len(sent)!=0 else ''}, {summary}"
        return img_prompt

    # Get the audio input from the user and return
    # the text from the audio
    def get_audio_input(self):
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
            audio = self.audio_recognizer.record(source)
            
        # Get the text from the audio
        with self.suppress_stdout():
            try:
                text = self.audio_recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                return ""
        
        return text

    # Get the response from GPT-3 and return the
    # response text
    def get_response_gpt(self, text, GPT_key):
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

    # Get a response from the other model and return the response text
    def get_response_other(self, text):
        # How many newlines are there?
        num = text.count("\n")

        # Get the model output. at the correct position
        output = self.other_text_model(text)[0]['generated_text'].split("\n")
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



    # Summarize the text so far.
    def summarize_text(self):
        # If the prompt is over the block size, save it
        # to memory. Summarization comes later
        if len(self.cur_prompt.split(" ")) > self.block_size:
            # Get a subset which of the block size
            splt = self.cur_prompt.split(" ")
            subset = " ".join(splt[:self.block_size]) + " "

            # The rest is the current prompt
            self.cur_prompt = " ".join(splt[self.block_size:])

            # Get the oldest item in the past output and clean it
            oldest_item = self.past_output[0]
            oldest_item = oldest_item.replace("Girlfriend: ", "").replace("Me: ", "").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")

            # Store the subset and move all subsets
            # up in the queue
            self.past_output = self.past_output[1:] + [subset]




            # If the oldest item is not "", summarize it
            # Summarize the subset
            if oldest_item != "":
                # Summarize it as the current summary
                self.past_summ = self.summarizer(
                    self.past_summ + "\n\n" + oldest_item,
                    min_length=16,
                    max_length=512,
                    no_repeat_ngram_size=3,
                    repetition_penalty=5.0,
                    num_beams=4, # Note: Over 4 beams and the model kills my computer
                    early_stopping=True,
                )[0]["summary_text"]
        
        # When saving is done, save files to disk
        self.disc_json = dict(
            past_summ=self.past_summ,
            past_output=self.past_output,
            cur_prompt=self.cur_prompt,
        )


    # Function to get a response and deal with the
    # response either from GPT or the other model
    def get_response(self, GPT_key=None):
        """
        The text used to respond is creafted upon
        all three components in the history.
        It will look like the following:
        [summary of the past]\n\n\n\n
        [saved prompts from the past][current prompt]
        """
        text = self.past_summ + "\n\n" +\
            "".join(self.past_output)+\
            self.cur_prompt

        # If the key is None, get a response from the
        # other model
        if GPT_key is None:
            resp = self.get_response_other(text)
        else:
            resp = self.get_response_gpt(text, GPT_key)

        # Sometimes a stupid output will be placed at the
        # beginning like [Random name]: [words].
        # let's remove these
        resp = resp.split(":")[-1].strip()

        # Add the new text to the prompt
        self.cur_prompt += f"Girlfriend: {resp}\n"

        # Before returning the respnse, we need to make sure
        # the text is being summarized
        self.summarize_text()

        # After the text has been update, update the
        # dictionary and save it
        self.disc_json["cur_prompt"] = self.cur_prompt
        json.dump(self.disc_json, open("config_file.json", "w"))

        # Return the response
        return resp


    # Given some text, generate a new image and return it
    def text_to_image(self, settings, characteristics, guidance_scale, text):
        # Get the image prompt
        img_prompt = self.build_img_prompt(text, settings, characteristics)
        
        # Get the image
        with self.suppress_stdout():
            with autocast("cuda"):
                image = self.imgGen(img_prompt, guidance_scale=guidance_scale)["images"]
                
        
        return image

    






    # Transcribes audio to text
    def audio_to_text(self, audio):
        try:
            text = self.recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            text = "" # Default to nothing
                
        return text

    # Overall function to generate text and audio. This
    # Function reads in the audio, transcribes it, and
    # starts the mouth movement threads.
    def generate_audio(self, custom_audio, custom_model, text, audio_pth, GPT_key):
        # Make sure a thread isn't already running
        if self.m_thread is not None:
            self.m_thread.join()
        
        # Get the audio if there is any
        if audio_pth:
            # Open the wav file and read in the data
            # Get the audio data
            audio = sr.AudioFile(audio_pth)
            with audio as source:
                audio = self.recognizer.record(source)
            
            text = self.audio_to_text(audio)
        
        # Add the text to the current prompt
        self.cur_prompt += f"Me: {text}\n"
        
        # Get the response
        if custom_model == True:
            ret_text = self.get_response()
        else:
            try:
                ret_text = self.get_response(GPT_key)
            except:
                gr.Error("GPT key is either invalid or not given")
                return "Error: GPT key is either invalid or not given."
        
        # Create audio and image for the returned text
        if len(ret_text) > 3:
            
            # Create the audio clip
            try:
                mixer.stop()
                mixer.music.unload()
            except pygame.error:
                pass
            self.create_audio(ret_text, custom_audio)
            
            # Start the mouth movement loop
            m_thread = threading.Thread(target=self.run_talk_loop, args=("tmp.mp3",))
            m_thread.start()
            
            # Save the text in case of errors
            self.last_text = ret_text
            
            return ret_text




    

    # Function used to generate images given text
    def generate_img(self, settings, characteristics, guidance_scale):
        # Generate an image from the current prompt
        ret_text = ""
        image = self.text_to_image(settings, characteristics, guidance_scale, ret_text)[0]
        
        # When an image is generated, load it in the animator
        old_add_movement = self.add_movement
        self.add_movement = False
        self.img_anim.load_new_image(img=image)
        self.add_movement = old_add_movement
        
        # Save the image in case of errors
        self.last_image = image
        
        # Ensure the image style vector is reset
        self.img_anim.pose *= 0
        
        # Force the image to be regenerated
        self.force_gen = True











    # Load the custom audio models
    def load_custom_audio(self, audio_model_path, audio_data_path):
        # Stuff for custom voice
        from Audio_Generation.Generation_Scripts.generation import Audio_Obj

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
    def create_audio(self, text, custom_audio):
        global audioObj
        if custom_audio:
            audioObj.synthesize(text)
            audioObj.vocode(play_audio=False)
        else:
            try:
                myobj = gTTS(text=text, lang='en', slow=False)
                myobj.save("tmp.mp3")
            except gTTSError:
                pass

    
    # Function to extract the word data from a mp3 file
    def extract_word_data(self, filename):
        # Make the audio a wav file
        f = AudioSegment.from_mp3(filename)
        f.export("tmp.wav", format="wav")
        
        # Read in the audio
        with wave.open("tmp.wav", "rb") as wf:
            # Prepare the model for rekognition
            rec = KaldiRecognizer(self.vosk_model, wf.getframerate())
            rec.SetWords(True)

            # get the list of JSON dictionaries
            results = []
            # recognize speech using vosk model
            data = wf.readframes(wf.getnframes())
            while len(data) > 0:
                if rec.AcceptWaveform(data):
                    part_result = json.loads(rec.Result())
                    results.append(part_result)
                data = wf.readframes(wf.getnframes())
            part_result = json.loads(rec.FinalResult())
            results.append(part_result)
        
        audio_trans = results[0]["result"]
        
        # Get the delay between each word
        for i in range(1, len(audio_trans)):
            audio_trans[i]["delay"] = audio_trans[i]["start"]-audio_trans[i-1]["end"]
        audio_trans[0]["delay"] = audio_trans[0]["start"]
        
        return audio_trans

    





    """
    The following code is used to run a thread to
    make the model blink every now and then.
    """

    # Literally all this function does is update the
    # eye part of the vector every so often
    async def blink_loop(self):
        # We want to iterate until a flag states
        # the animation should stop
        while not self.stop_animating:
            # Wait a little to blink again
            if self.img_anim.eye_cycle_end:
                # Blink anywhere between 2 and 7 secods with
                # a mean around 5 seconds (avg blink wait time)
                t = np.clip(np.random.normal(5, 1, size=1)[0], 2, 7)

                # Wait a little before blinking again
                time.sleep(t)
                self.img_anim.eye_cycle_end = False
            
            # Update the vector
            self.img_anim.Move_eyes()
            
            # Wait for a new frame to be generated
            while self.img_anim.eye_frame_disp == False:
                time.sleep(0.001)

    # Used to make a thread running the blink loop
    def run_blink_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        self.stop_animating = False

        loop.run_until_complete(self.blink_loop())
        loop.close()





    


    """
    Like with the blink loop, the below code is used to
    run a thread for making the imag talk
    """
    async def talk_loop(self, filename):
        # Get the audio transcript
        audio_trans = self.extract_word_data(filename)
        
        # Play the audio
        try:
            mixer.init()
            mixer.stop()
            mixer.music.unload()
            try:
                mixer.music.load(filename)
                mixer.music.play()
            except pygame.error:
                s = mixer.Sound(filename)
                s.play()
        except pygame.error:
            from IPython.display import Audio, display, clear_output
            clear_output(wait=True)
            display(Audio(filename, autoplay=True))
        
        # Iterate over all parts of the audio transcription
        for idx, part in enumerate(audio_trans):
            # Get the beginning and end of the audio piece
            start = part["start"]
            end = part["end"]
            delay = part["delay"]
            
            # Wait for the next audio part according to the
            # delay in the audio. This delay should also take
            # into account the expected generation time of the
            # image as the delay starts after the previous generation
            if idx != 0:
                delay = max(0, delay-self.img_anim.EMA)
            time.sleep(delay)
            
            # Get the entire audio clip length
            length = end-start
            
            # Setup the mouth movement cycle
            self.img_anim.setup_mouth_movement(length)
            
            # Mouth movement is being generated
            self.generating_mouth_movement = True
            
            # Iterate until the movement is done for this part
            while self.img_anim.mouth_cycle_end == False:
                # Update the vector
                self.img_anim.Move_mouth()
                
                # Wait for a new frame to be generated
                while self.img_anim.mouth_frame_disp == False:
                    time.sleep(0.001)
                    
            # Mouth movement is not being generated
            self.generating_mouth_movement = False
        
        
    # Used to make a thread running the talk loop
    def run_talk_loop(self, filename):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        loop.run_until_complete(self.talk_loop(filename))
        loop.close()





    """
    Main event loop that moves the stored image when events are
    triggered on either talking threads or blinking threads
    """
    def event_loop(self):
        # Initial update to make everything visible
        yield self.last_image, gr.update(visible=True),\
                gr.update(visible=False)
        
        # Quick calibration. Blink 10 times
        # and calibrate the time it takes
        # to show the image for the EMA
        for i in range(0, 10):
            s = time.time()
            self.img_anim.eye_cycle_end = False
            while self.img_anim.eye_cycle_end == False:
                self.img_anim.Move_eyes()
                img = self.img_anim.change_pose()
                self.img_anim.update_EMA(time.time()-s)
                s = time.time()
                yield img, gr.update(), gr.update()
        self.img_anim.eye_cycle_end = False
        
        # Start the blink loop
        if self.b_thread == None:
            self.b_thread = threading.Thread(target=self.run_blink_loop, args=())
            self.b_thread.start()
        
        while True:
            # If the image is forced to be reloaded, generate
            # the image and reset the flag
            if self.force_gen == True:
                img = self.img_anim.change_pose()
                self.force_gen = False
                yield img, gr.update(), gr.update()
            
            # If movement shouldn't be added, skip the loop
            if self.add_movement == False:
                time.sleep(0.0001)
                continue
            
            # Wait until a new frame needs to be generated
            if self.generating_mouth_movement == True:
                if self.img_anim.mouth_frame_disp == False:
                    # Change the pose and show the image
                    img = self.img_anim.change_pose()

                    yield img, gr.update(), gr.update()
            else:
                # Start the mouth movement loop
                if self.img_anim.eye_frame_disp == False:
                    # Change the pose and show the image
                    img = self.img_anim.change_pose()

                    yield img, gr.update(), gr.update()
            
            time.sleep(0.0001)





    # Reset the memory of the model
    def reset_memory(self):
        self.past_summ = deepcopy(self.initial_summary)
        self.past_output = ["" for i in range(self.num_blocks)]
        self.cur_prompt = deepcopy(self.initial_prompt)



    # Used to change the blink time given a new rate. This value
    # is limited between 0.5 and 2
    def change_blink_time(self, new_blink_time):
        min_val = 0.5
        max_val = 2.0

        # Limit the blink time
        new_blink_time = min(max_val, max(min_val, new_blink_time))

        # Change the blink time
        self.img_anim.total_blink_time_i = new_blink_time
        self.img_anim.total_blink_time = new_blink_time
        self.img_anim.eye_num_frames = (self.img_anim.total_blink_time//self.img_anim.EMA)
        self.img_anim.eye_midpoint = max(1, round(math.ceil(self.img_anim.eye_num_frames/2)))
