# Imports
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import numpy as np
import PIL
from PIL import Image
import gradio as gr
import sys
import openai
import speech_recognition as sr
import pygame
from pygame import mixer
import cv2
from vosk import KaldiRecognizer, SetLogLevel
from vosk import Model as vosk_Model
from pydub import AudioSegment
import wave
import json
from Img_Mover.Img_Mover import Img_Mover
from Girlfriend_Obj import Girlfriend_Obj
import matplotlib.pyplot as plt
import multiprocess
import gradio as gr
import time
from copy import deepcopy
import asyncio
import threading
import os

# # Path to the custom audio model
# audio_model_path = "Audio_Generation/Generation_Scripts/saved_models/default"

# # Path to the custom audio data
# audio_data_path = "Audio_Generation/Generation_Scripts/data/albedo"

# Path to the custom model to load in
custom_model_path = "CustomModel/"

# # The initial summary is initially a basic prompt telling GPT-3 who it is
# initial_summ = "You are my female waifu girlfriend who loves me."
# # The initial prompt tells GPT-3 how to respond
# initial_prompt = "Me: Hi\nYou: Hello\n\n"\
#     "Me: How are you?\nYou: Good. How are you?\n\n"\
#     "Me: I'm good.\nYou: Nice to meet you.\n\n"

initial_summ = "The following is a conversation with me and my waifu girlfriend\n\n"
initial_prompt = "Me: Hello\nGirlfriend: Hello\n"\
         "Me: How are you?\nGirlfriend: I am good\n"

# Setup function to setup the environment
# memory_file = "config_file.json"
memory_file = None
MyGirlfriend = Girlfriend_Obj(initial_summ, initial_prompt, False, custom_model_path=custom_model_path, saved_memory=memory_file)

# Device must be cuda
device = torch.device("cuda:0")

def audio_auto_submit(custom_audio, custom_model, text, audio_pth, GPT_key):
    if audio_pth != None:
        return MyGirlfriend.generate_audio(custom_audio, custom_model, text, audio_pth, GPT_key)
    return MyGirlfriend.last_text

# Initialize the audio mixer
mixer.init()
mixer.music.unload()
    
# Handle changes to the motion switch which either turns on or
# off image motion
def handle_motion_switch(switch_value):
    MyGirlfriend.add_movement = switch_value
    
    # Ensure the image is in the default position
    MyGirlfriend.img_anim.pose *= 0
        
    # Force a reload in the image
    MyGirlfriend.force_gen = True
    
# Handles file uploads
def upload_file(file):    
    # Load the image as a PIL object
    image = Image.open(file.name)
    
    # When an image is generated, load it in the animator
    old_add_movement = MyGirlfriend.add_movement
    MyGirlfriend.add_movement = False
    MyGirlfriend.img_anim.load_new_image(img=image)
    MyGirlfriend.add_movement = old_add_movement
    
    # Save the image in case of errors
    MyGirlfriend.last_image = image
    
    # Ensure the image style vector is reset
    MyGirlfriend.img_anim.pose *= 0
    
    # Force the image to be regenerated
    MyGirlfriend.force_gen = True
    
    return file.name
    
# Handles image saving
def save_img():
    if not os.path.exists("saved_images"):
        os.mkdir("saved_images")
    filename = fr"./saved_images/{time.ctime().replace(' ', '-').replace(':', '.')}.png"
    if type(MyGirlfriend.last_image) is not PIL.Image.Image:
        Image.fromarray(MyGirlfriend.last_image.clip(0, 255).astype(np.uint8)).save(filename)
    else:
        MyGirlfriend.last_image.save(filename)
        
# Function used to test the mouth movement
def test_mouth():
    # Make sure the mouth isn't already moving
    if MyGirlfriend.generating_mouth_movement == True:
        return
    
    # Make sure the thread is not running
    if MyGirlfriend.m_thread is not None:
        MyGirlfriend.m_thread.join()
    
    # Start the mouth movement loop
    MyGirlfriend.m_thread = threading.Thread(target=MyGirlfriend.run_talk_loop, args=("test_audio.mp3",))
    MyGirlfriend.m_thread.start()
    
# Loads a memory file into the model
def load_mem(file):
    # Get the filename
    filename = file.name
    
    # Load in the file and upload the memory
    outTxt = MyGirlfriend.load_mem(filename)
    
    return filename, outTxt
    

interface = gr.Blocks(css="#color_red {background-color: #f44336}")
with interface:
    with gr.Tabs():
        with gr.TabItem("Intro"):
            gr.Textbox("""
            Below is an intro explaining how this app works...
            
            Generation Tab:
              Before starting, make sure to click the "Setup interface" button to setup the inferface and to begin using the app.
              
              The upper-most part of the interface includes two tabs: "Voice-based Chat" and "Text-based Chat" which are used to repond to the AI. Voice-based allows you to use your mic to talk to the AI while text-based allows you to chat with the AI using text. The audio is auto-submitted for response while the text requires either pressing the "enter" key or clicking the "Generate Audio" button.
              
              The next part is the "Response" text field. The latest response the AI gave will appear here.
              
              Below reponse is a section split into two parts. The left-most part is the currently generated image. The rightmost section has multiple parts:
              1. "Add motion to image?" checkbox is used to toggle image animation. If checked, the image will be animated. The animation includes blinking and mouth movement assuming the image is in the correct form.
              2. "Mouth movement test" can be used to check if mouth movement works for the current image.
              3. "Save current image" saves the currently generated image to a folder named "saved_images". The filename will be the current time and date so that images don't overwrite eachother
              4. "Upload an image" is used to upload an image you want to load in as opposed to generating one until one looks good. Clikcing on this button allows you to select the image you want to display.
              
              At the bottom of this section, there are two buttons: "Generate Audio" and "Generate Image". "Generate Audio" takes the currently entered text and generates a new response from the AI. "Generate Image" is used to generate a new image and display it.
                 
              Some notes about image animation:
                The image must be in the correct form to be animated correctly. The image should be a face-shot photo to ensure that blinking is done correctly. Mouth movement will occur if the image is face forward and when audio is generated. Sometimes the mouth movement doesn't work and if this is the case, you should probably just generate an image until movement works.
            
            
            Settings Tab:
              The settings tab has several uses from loading in past memories to changing the style of the image to generate.
              
              The first block in this tab is the "Use custom chat model?" checkbox. If this box is checked, a free custom model will be used to respond. Otherwise GPT-3 will respond. If the box is unchecked, an OpenAI key is required which can obtained following this article: https://elephas.app/blog/how-to-create-openai-api-keys-cl5c4f21d281431po7k8fgyol0 If a key isn't provided, an error will be shown in place of the response text.
              
              The next block is the "Settings" blocks which is used to setup the style of the image and how it's generated. Settings can be found at the following link (though do be warned, the site has some sus images, not my doing btw): https://danbooru.donmai.us/wiki_pages/tag_group:image_composition
              
              The next block is "Characteristics" which is also used to style the generated images. These prompts are more of how you want the generated image to look like. Should it be female or male? What color hair?
              
              The "settings" block and "characteristics" block actually have no difference when implemented, but it's nice to break up the difference between image settings and image characteristics.
              
              The next block is the "Guidance value" which is used as a tradeoff between Fidelity (how good the image looks) and variance (kind of how creative the model is). A value of 1 is required, and having a value too high will cause garbage to be produced. Keeping this value around 10 seems to work well.
              
              The "blink time" button and field allows you to change the number of seconds it takes to make a full blink.
              
              The next part is a memory loading system. As the conversation goes on, the conversation is saved to a memory file called "config_file.json". This file can be loaded back in through this section of the settings to replace the current conversation with a past one saved in a .json file. The text box next to the upload button signals where the upload was a success or a failure.  
              
              The last part is a reset button to reset the current memory to the initial prompt.
            """)
        
        with gr.TabItem("Generation"):
            gen_col = gr.Column(visible=False)
            with gen_col:
                # Talking to the AI
                with gr.Tabs():
                    with gr.TabItem("Voice-based Chat"):
                        audio = gr.Audio(source="microphone", type="filepath", label="Response", live=True)
                    with gr.TabItem("Text-based Chat"):
                        text = gr.Textbox(label="Text", value="I love you!", interactive=True)
                response = gr.Textbox(label="Response", value="", interactive=False)

                with gr.Row():
                    # Note gallery expects a 3-D array: (L, W, 3)
                    gallery = gr.Image(label="Generated images", show_label=False)\
                        .style(height=512)

                    with gr.Column():
                        # Switch to generate a new image with audio or keep the
                        # image static
                        motion_switch = gr.Checkbox(value=True, label="Add motion to image?")
                        motion_switch.change(fn=handle_motion_switch, inputs=[motion_switch], outputs=[])

                        # Button to test mouth movement
                        btn_mouth_test = gr.Button("Mouth movement test")
                        btn_mouth_test.click(fn=test_mouth, inputs=[], outputs=[])

                        # Button to save the currently generated image
                        btn_save_img = gr.Button("Save Current Image")
                        btn_save_img.click(fn=save_img, inputs=[], outputs=[])

                        # Button to load an image
                        upload_button = gr.UploadButton("Upload an image", file_types=["image"], file_count="single")
                        upload_button.upload(fn=upload_file, inputs=[upload_button])

                with gr.Row():
                    # Button to generate new audio
                    btn_audio = gr.Button("Generate Audio")

                    # Button to generate new audio
                    btn_img = gr.Button("Generate Image")
            
            # Button to load and setup the generation tab
            btn_load = gr.Button("Setup interface")
            btn_load.click(fn=MyGirlfriend.event_loop, inputs=[], outputs=[gallery, gen_col, btn_load], queue=True)
            
            
        with gr.TabItem("Settings"):
            # Switched for which model to use
            custom_model = gr.Checkbox(value=True, label="Use custom chat model? (False to use GPT, True to use custom model)")
            GPT_key_ = gr.Textbox(label="Key to use GPT-3 (if using GPT-3)\nNote: If you don't have one go here: https://elephas.app/blog/how-to-create-openai-api-keys-cl5c4f21d281431po7k8fgyol0", value="", interactive=True)
            custom_audio = gr.Checkbox(value=False, label="Use custom audio model?")

            # Settings for the image
            settings = gr.Textbox(label="Settings", value= "1girl,solo focus,very wide shot,feamle focus,ratio:16:9,detailed,looking at viewer,facing viewer,facing forward,vtuber", interactive=True)
            characteristics = gr.Textbox(label="Characteristics", value="waifu,female,brown hair,blue eyes,sidelocks,slight blush,happy", interactive=True)
            guidance_scale = gr.Number(label="Guidance value - Tradeoff between creativity and image fidelity (greater than 1.0)", value=10.0, interactive=True, precision=1)
            with gr.Row():
                blink_time = gr.Number(label="Time for a full blink (in seconds) (limited between 0.5 and 2.0)", value=0.6, interactive=True, precision=2)
                blink_time_btn = gr.Button(value="Change blink time").click(MyGirlfriend.change_blink_time, inputs=[blink_time])
                blink_time.submit(MyGirlfriend.change_blink_time, inputs=[blink_time])
            
            # Used to load a memory file
            with gr.Column():
                with gr.Row():
                    trash_file_output = gr.File(visible=False)
                    mem_load_btn = gr.UploadButton("Load memory file", file_types=["json"], file_count="single")
                    mem_file_success = gr.Textbox(label="Was the load successful?", value= "", interactive=False)
                    mem_load_btn.upload(fn=load_mem, inputs=[mem_load_btn], outputs=[trash_file_output, mem_file_success])
            
            # Used to treset the memory of the model
            reset_btn = gr.Button(value="Reset Memory", elem_id="color_red")
            reset_btn.click(MyGirlfriend.reset_memory, inputs=[], outputs=[])
            
        # When the audio is changed, we want to auto submit it
        audio.change(fn=audio_auto_submit, inputs=[custom_audio, custom_model, text, audio, GPT_key_], outputs=[response])
        
        # When the button or text is submitted, we want to generate new audio
        btn_audio.click(fn=MyGirlfriend.generate_audio, inputs=[custom_audio, custom_model, text, audio, GPT_key_], outputs=[response])
        text.submit(fn=MyGirlfriend.generate_audio, inputs=[custom_audio, custom_model, text, audio, GPT_key_], outputs=[response])
        
        # When the image button is clicked, we want to generate a new image
        btn_img.click(fn=MyGirlfriend.generate_img, inputs=[settings, characteristics, guidance_scale], outputs=[])

interface.queue(concurrency_count=3).launch(debug=False, share=False)