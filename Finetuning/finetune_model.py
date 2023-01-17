import os
import random
from transformers import AutoTokenizer
from transformers import TextDataset,DataCollatorForLanguageModeling
import torch
from transformers import Trainer, TrainingArguments, AutoModelWithLMHead, AutoModelForCausalLM
from transformers import pipeline


# Base model to finetune
base_model = "EleutherAI/gpt-neo-1.3B"

# What percent of the data is text data
test_per = 0.1

# Files to download/load data to
train_file_name = f"Finetuning{os.sep}train_data_mini.txt"
test_file_name = f"Finetuning{os.sep}test_data_mini.txt"





# Load in the model
tokenizer = AutoTokenizer.from_pretrained(base_model, framework="pt", device=torch.device("cpu"), torch_dtype=torch.float16)

# Get the tokenizer max size
max_size = 2048





# Create the dataset
def load_dataset(train_path,test_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=max_size)

    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=max_size)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, # non masking for generation
    )
    return train_dataset,test_dataset,data_collator

train_dataset,test_dataset,data_collator = load_dataset(train_file_name,test_file_name,tokenizer)

# Setup the model trainer
model = AutoModelForCausalLM.from_pretrained(base_model).to(torch.device("cpu"))

training_args = TrainingArguments(
    output_dir="Finetuning/outputs", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=3, # number of training epochs
    per_device_train_batch_size=1, # batch size for training
    per_device_eval_batch_size=1,  # batch size for evaluation
    eval_steps = 400, # Number of update steps between two evaluations.
    save_steps=800, # after # steps model is saved
    warmup_steps=500,# number of warmup steps for learning rate scheduler
    no_cuda=False,
    fp16=True,
    fp16_full_eval=True,
    gradient_accumulation_steps =2
    )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model()

# Test the model
test_model = pipeline('text-generation',model="Finetuning/outputs", tokenizer='EleutherAI/gpt-neo-1.3B')
print(test_model('I love you')[0]['generated_text'])