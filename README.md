# Summary
This repo contains code for an AI girlfriend which can be run using the "main.ipynb" files in the root directory.
All other files are either support or depreciated.

The depreciated directory contains files I worked on to get to the final version of the repository.
I thought it may contains some useful info and I decided not to delete it.

The colab that goes along with this repository can be found [here](https://colab.research.google.com/drive/1Nl5ioIkJdrsE-IoMUNPMsDt-wMi18JLN?usp=sharing)

The medium article going along with this repository can be found [here](https://gmongaras.medium.com/coding-a-virtual-ai-girlfriend-f951e648aa46)

This repo has two spilts:
1. [A reduced version for cloning purposes](https://github.com/gmongaras/AI_Girlfriend_Reduced)
2. [A very reduced version for the article I wrote](https://github.com/gmongaras/AI_Girlfriend_Medium)


To run the code from this repo (GPU required):
1. Clone the repo
2. Start a terminal in the root directory of this repo
3. type `jupyter lab`
4. Open `main.ipynb` and run the cells. The topmost cell can be uncommented to download the necessary packages and the versions that worked on my machine.


# Features
Generation tab:
1. You can speak to her using the "Record From Audio" button.
2. If you are shy, you can enter text into the textbox titled "Text-based Chat" on the "Generation" page. You can press "Enter" while the cursor is in this box to submit the text for a response.
3. Using either the chat or audio replies, the AI girlfriend will reply and text will popup in the "Reponse" section. Additionally, she will speak the text and try her best to lip-sync it.
4. The "Generate Audio" button submits either the text or audio submission from the user to the AI girlfriend so she can reply. This button is kind of useless since entering audio or pressing "Enter" suto-submits.
5. The "Generate Image" button generates a new image and displays it in the image frame. The image is generated using the promtps entered on the "Settings" page.
6. A box can be checked or unchecked to turn on or off the animation.
7. "Mouth Movement Test" plays a small audio clip to test if the mouth movement works.
8. "Save current image" saves the currently display image to a new folder named "saved_images"
9. "Upload an image" allows the user to upload an image to replace the current one in frame.

Settings tab:
1. The top-most box and check box allow the user to toggle on/off GPT-3 as the response bot. If GPT-3 is used, then a key must be provided in the box below the check box.
2. A "Settings" box including settings for the image. The settings can be found at this link (btw there is some bad content on there, not my doing :/): [https://danbooru.donmai.us/wiki_pages/tag_group:image_composition](https://danbooru.donmai.us/wiki_pages/tag_group:image_composition)
3. A "Characteristics" box including characteristics to add to the image.
4. A "Guidance Value" box with a single floating point number. This number tells the model how much to use classifier guidance. The higher the number, the better the image usually looks, but the less creative the model is (up to a point). This value is essentially a tradeoff between variance (low) and fidelity (high).
5. A setting to change the blink time. This value can be between 0.5 and 2 seconds and times how long the average blink should take. A time of 2 seconds takes the AI girlfriend 2 second to blink once.
6. A "Load memory" area with a textbox and button. Click the button to upload a memory file. THe textbox states whether this upload was a success or not.
7. A "Reset Memory" button which resets the memory of the AI girlfriend completely.

Passive:
- As the conversation goes on, a file named "config_file.json" is generated. This file stores the memory of the current AI girlfriend so it can be loaded if needed.
- As the conversation goes on, it is summarized in the background to save memory and allow for an infinite coversation while trying to retain a memory for the AI girlfriend.


# Example Screenshots
![image](https://user-images.githubusercontent.com/43501738/216734158-afe1769a-ffaf-481d-8c5d-97a24d7e85c3.png)
![image](https://user-images.githubusercontent.com/43501738/216734173-0a1cef86-6463-4690-bb5b-c72dd4a52b01.png)


# Example Demo
[![Video Demo](https://img.youtube.com/vi/PxsyIjzlcCM/0.jpg)](https://www.youtube.com/watch?v=PxsyIjzlcCM)
