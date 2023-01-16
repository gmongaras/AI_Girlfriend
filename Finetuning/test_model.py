from transformers import pipeline



# Test the model
test_model = pipeline('text-generation',model="Finetuning/outputs", tokenizer='EleutherAI/gpt-neo-1.3B', max_new_tokens=25)


num = 3 # Number of newlines to track new output
prompt = "Person: Hi\nYou: Hello\n\n" # Initial prompt
while True:
    # Get user input
    print("Prompt: ", end="")
    text = input()

    if text == "":
        break

    # Add the text to the current prompt
    text = f"Person: {text}\nYou: "
    prompt += text
    num += 1

    # Get the model output. at the correct position
    output = test_model(prompt)[0]['generated_text'].split("\n")
    output_new = output[num].strip()

    # Make sure the output is not blank
    tmp = 1
    output_new = output_new.replace("You:", "").replace("Person:", "")
    while output_new == "":
        output_new = output[num+tmp].strip()
        tmp += 1

    # Print the output
    output_new = output_new.replace("You:", "").replace("Person:", "")
    print(output_new)

    # Add the outputted text to the prompt
    prompt += output_new.strip() + "\n\n"

    # How many newlines are there?
    num = prompt.count("\n")