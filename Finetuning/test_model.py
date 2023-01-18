from transformers import pipeline
import torch
from string import punctuation



# Test the model
test_model = pipeline('text-generation',model="outputs/r/",
                      tokenizer='EleutherAI/gpt-neo-1.3B',max_new_tokens=50,
                      torch_dtype=torch.float16,framework="pt",
                      device=torch.device("cuda:0"))


# Initial prompt
prompt = "Hi\nHello\n\n"\
         "How are you?\nGood. How are you?\n\n"\
         "I'm good.\nNice to meet you.\n\n"\

# Number of prompt entered
num_prompts = 0

# Number of prompt until output can be extended
num_till_exten = 5

while True:
    # How many newlines are there?
    num = prompt.count("\n")

    # Get user input
    print("Prompt: ", end="")
    text = input()

    if text == "":
        break

    # Add the text to the current prompt
    text = f"{text}\n"
    prompt += text
    num += 1

    # Get the model output. at the correct position
    output = test_model(prompt)[0]['generated_text'].split("\n")
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
    if num_till_exten <= num_prompts:
        more_max = 1 # Max limit on how much more to add
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

    # Print the output
    #output_new = output_new.replace("You:", "").replace("Person:", "")
    print(cur_out)

    # Add the outputted text to the prompt
    prompt += cur_out.strip() + "\n\n"
    num_prompts += 1