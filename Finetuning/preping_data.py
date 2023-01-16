# Prep data to be in two files
import os
import random



# What percent of the data is text data
test_per = 0.1

# Directories to the data
data_dir = f"Finetuning{os.sep}data_clean"
data_files = [str(i)+".txt" for i in range(1, 11)]

# Files to download/load data to
train_file_name = f"Finetuning{os.sep}train_data_mini.txt"
test_file_name = f"Finetuning{os.sep}test_data_mini.txt"

# test/train data sizes
test_size = 0
train_size = 0

# Open the output files
with open(train_file_name, "w") as train_file:
    with open(test_file_name, "w") as test_file:

        # Iterate over each file and load in the data
        for file in data_files:
            # Open the file
            with open(data_dir + os.sep + file, "r") as f:
                # Iterate over all data in the file
                for line in f:
                    # Get a random number between 0 and 1
                    num = random.uniform(0, 1)
                    
                    # If the number is greater than the test size,
                    # add it to the train data
                    if num > test_per:
                        train_file.write(line)
                        train_size += 1
                    else:
                        test_file.write(line)
                        test_size += 1
                        
print(f"Number of train data: {train_size}")
print(f"Number of test data: {test_size}")