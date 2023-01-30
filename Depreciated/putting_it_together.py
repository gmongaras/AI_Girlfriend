import openai
from gtts import gTTS
import pygame
from pygame import mixer
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from string import punctuation
from heapq import nlargest
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
import asyncio
import json
from Talking_Head.Talking_Head import Talking_Head
import multiprocess




# Test the custom audio
def test_audio(audio_model_path, audio_data_path):
    # Stuff for custom voice
    from Audio_Generation.Generation_Scripts.generation import Audio_Obj

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







class WaifuObj:
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
        # Puncuation tokenizer
        self.tokenizer = RegexpTokenizer(r'\w+')

        # VADER sentiment analyzer
        self.sent_model = SentimentIntensityAnalyzer()

        # Audio recognizer
        self.audio_recognizer = sr.Recognizer()



        # Initialize the summary model
        self.summ_model = KeyBERT()


        

        # Image generation model
        print("Initializing image model...")
        self.imgGen = StableDiffusionPipeline.from_pretrained(
            'hakurei/waifu-diffusion',
            torch_dtype=torch.float16,
            cache_dir="D:/python-libs/hugging-face-cache",
        ).to('cuda')
        # Remove filter
        self.imgGen.safety_checker = lambda images, clip_input: (images, False)
        print("Image model initialized!")



        # Get the image generation model if
        # the GPT model is notused
        # https://huggingface.co/EleutherAI/gpt-neo-1.3B/
        # Max len is 2048
        print("Initializing custom text model")
        self.other_text_model = pipeline('text-generation',model=custom_model_path,
                        tokenizer='EleutherAI/gpt-neo-1.3B',max_new_tokens=50,
                        torch_dtype=torch.float16,framework="pt",
                        device=torch.device("cuda:0"),
                        pad_token_id=50256)
        print("Custom text model initialized!")
        # Otherwise, use the GPT model


        # Load in the large summarizer model
        # https://huggingface.co/pszemraj/led-large-book-summary
        self.summarizer = pipeline(
            "summarization",
            'pszemraj/led-large-book-summary',
            device=0 if torch.cuda.is_available() else -1,
            framework="pt",
            torch_dtype=torch.float16,
        )

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

    # Get the sentiment of text
    def get_sent(self, text):
        sents = self.sent_model.polarity_scores(text)
        
        if sents["neg"] > 0.5:
            return "angry"
        elif sents["pos"] > 0.5:
            return "happy"
        else:
            return ""


    # Summary function for a single line
    def summarize_single(self, text):
        # Get the keywords
        keywords = self.summ_model.extract_keywords(text)

        # Get keywords above a threshold
        words = ", ".join([word[0] for word in keywords])

        return words


    # Get the summary of the text
    def get_summ(self, text):
        # Get the summary
        summary = self.summarize_single(text)
        
        # Remove stopwords and puncuation from the summary
        filtered = [word for word in self.tokenizer.tokenize(summary) if word not in stopwords.words('english')]
        
        return " ".join(filtered)

    # Build a prompt for the image
    def build_img_prompt(self, text, settings, characteristics):
        # Get the summary and sentiment
        sent = self.get_sent(text)
        summary = self.get_summ(text)
        
        # Create the image prompt
        # settings = "1girl, very wide shot, simple background, solo focus, female focus, looking at viewer, ratio:16:9, detailed"
        # characteristics = "waifu, female, brown hair, blue eyes, sidelocks, slight blush, fox ears"
        # sent = "furious"
        # summary = "'I hope get know better' to viewer"
        prompt = f"{settings}, {characteristics}, {sent+', ' if sent != '' else ''}"#'{summary}' to viewer"
        return prompt

    # Get the audio input from the user
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

    # Get the response from GPT-3
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

    # Get a response from the other model
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
            oldest_item = oldest_item.replace("You: ", "").replace("Me: ", "").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")

            # Store the subset and move all subsets
            # up in the queue
            self.past_output = self.past_output[1:] + [subset]




            # If the oldest item is not "", summarize it
            # Summarize the subset
            if oldest_item != "":
                # Summarize it as the current summary
                self.past_summ = self.summarizer(
                    self.past_summ + "\n\n\n\n" + oldest_item,
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
    # response
    def get_response(self, GPT_key=None):
        """
        The text used to respond is creafted upon
        all three components in the history.
        It will look like the following:
        [summary of the past]\n\n\n\n
        [saved prompts from the past][current prompt]
        """
        text = self.past_summ + "\n\n\n\n" +\
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
        self.cur_prompt += f"You: {resp}\n\n"

        # Before returning the respnse, we need to make sure
        # the text is being summarized
        self.summarize_text()

        # After the text has been update, update the
        # dictionary and save it
        self.disc_json["cur_prompt"] = self.cur_prompt
        json.dump(self.disc_json, open("config_file.json", "w"))

        # Return the response
        return resp



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
            myobj = gTTS(text=text, lang='en', slow=False)
            myobj.save("tmp.mp3")


# This function is used to thread the blinking animation
# of the generated image
def blink_loop(img, fig, ax):
    from matplotlib import animation
    import numpy as np
    import matplotlib.pyplot as plt
    plt.axis('off')

    # Initialize the Talking Head class to add movement to images
    img_anim = Talking_Head(torch.device("cuda:0"), 0.60)

    # Load in the new image
    img_anim.load_new_image(img=img)

    # Default pose
    img = img_anim.change_pose()
    im = ax.imshow(img, animated=True)

    # Update loop function
    def update_image(i):
        # Update the vector
        img_anim.Move_eyes()
        
        # Change the pose
        img = img_anim.change_pose()
        im.set_array(img)

        # Wait a little to blink again
        if img_anim.cycle_end:
            # Blink anywhere between 2 and 7 secods with
            # a mean around 5 seconds (avg blink wait time)
            t = np.clip(np.random.normal(5, 1, size=1)[0], 2, 7)

            plt.pause(t)
            img_anim.cycle_end = False

    ani = animation.FuncAnimation(fig, update_image, interval=0)
    plt.show()




# Main loop that uses the object
def main(obj, custom_audio, custom_model, img_settings, img_characteristics, guidance_scale, GPT_key):
    """
    We only need keyboard info in main

    The reason the imports are here is so this script can
    be run independently from google colab, but
    google colab can still use these functions. Google
    colab dies when trying to use these function, so
    I'm just not going to use them
    """
    from pynput.keyboard import Key, Listener

    # Is enter or space pressed? Currently
    # this is False
    global enter_pressed
    global space_pressed
    enter_pressed = False
    space_pressed = False

    # Key capture functions
    global on_press
    global on_release
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

    # Used to collect keyboard events
    listener = Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()


    # Like with the previous one, this module is only
    # availble on windows, so I'm just going to import
    # it here so colab can actually run
    import msvcrt





    while True:
        # Wait for person to press space
        print("Press space to talk to my waifu")
        while space_pressed == False:
            pass
        space_pressed = True
        print("Press enter when done speaking")
        
        # Get the audio input
        text_prompt = obj.get_audio_input()
        
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
        obj.cur_prompt += f"Me: {text_prompt}\n"
        
        # Get the text from the model
        if custom_model == True:
            ret_text = obj.get_response()
        else:
            ret_text = obj.get_response(GPT_key)
        
        # Create audio and the image for the returned text
        if len(ret_text) > 3:
            # Create the audio clip
            pygame.mixer.stop()
            mixer.music.unload()
            obj.create_audio(ret_text, custom_audio)
            
            # Play the audio
            try:
                mixer.music.load('tmp.mp3')
                mixer.music.play()
            except pygame.error:
                s = mixer.Sound('tmp.mp3')
                s.play()
            
            # Get the image prompt
            img_prompt = obj.build_img_prompt(ret_text, img_settings, img_characteristics)
            
            # Get the image
            with obj.suppress_stdout():
                with autocast("cuda"):
                    image = obj.imgGen(img_prompt, guidance_scale=guidance_scale)["images"][0]
            

            # When the image is generated, we need to animate it. We
            # will do so on another thread and kill the old
            # one if it exists
            if obj.anim_thread != None:
                obj.anim_thread.terminate()
            # Create the figure
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1) 

            obj.anim_thread = multiprocess.Process(target=blink_loop, args=(image, fig, ax))
            obj.anim_thread.daemon = True
            obj.anim_thread.start()

            # # Show the image
            # fig, ax = plt.subplots()
            # fig.subplots_adjust(0,0,1,1)
            # ax.set_axis_off()
            # ax.imshow(image)
            # plt.show()
            # del image







if __name__=="__main__":
    # The initial summary is initially a basic prompt telling GPT-3 who it is
    initial_summ = "You are my female waifu girlfriend who loves me."\
    # The initial prompt tells GPT-3 how to respond
    initial_prompt = "Me: Hi\nYou: Hello\n"\
        "Me: How are you?\nYou: Good. How are you?\n"\
        "Me: I'm good.\nYou: Nice to meet you.\n"

    # Use custom audio or not
    custom_audio = False

    # Use custom model or GPT3
    custom_model = True

    # Path to the custom audio model
    audio_model_path = "Audio_Generation/Generation_Scripts/saved_models/default"

    # Path to the custom audio data
    audio_data_path = "Audio_Generation/Generation_Scripts/data/albedo"

    # Path to the custom model to load in
    custom_model_path = "Finetuning/outputs/r/"

    # Settings and characteristics for the output image
    # img_settings = "1girl, very wide shot, simple background, solo focus, female focus, looking at viewer, ratio:16:9, detailed"
    img_settings = "1girl, very wide shot, solo focus, feamle focus, ratio:16:9, detailed, looking at viewer, facing viewer, facing forward, vtuber, pure black background, chest and head"
    img_characteristics = "waifu, female, brown hair, blue eyes, sidelocks, slight blush, fox ears"
    guidance_scale = 10.0
    
    # Setup the interface
    memory_file = None#"config_file.json"
    obj = WaifuObj(initial_summ, initial_prompt, False, audio_model_path, audio_data_path, custom_model_path, memory_file)
    # setup(False, audio_model_path, audio_data_path, custom_model_path)
    #test_audio(audio_model_path, audio_data_path)

    # Run the interface
    GPT_key = "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"
    main(obj, custom_audio, custom_model, img_settings, img_characteristics, guidance_scale, GPT_key)