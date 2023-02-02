import pysubparser
from pysubparser import parser
import os
import re







# I only feel like dealing with these extensions
# Known I want to deal with:
#   ass - Some sort of CSV type encoding
#   ssa - Same as ass
#   srt - Some sort of chunk separated format
# Known not dealing with:
#   sml - Looks like some weird xml-like format, but I
#         don't want to deal with it
#   xml - Probably not going to have any useful data
#   ttf - This is a text style file
#   otf - Same as ttf
#   txt - Usually has nothing of value
#   idx/sub - Stupid files that go together to make
#             subtitles, but are annoying to actually
#             get plain-text data from
#   jpg/png - The heck?
extens = set(["ass", "ssa", "srt"])


dir_to_look_at = "gathering_data"
thresh = 100000



# Checks if a string is trash or not
def is_trash(string):
    # If the first, second or third character is a number, we can
    # probably remove it
    if len(string) < 3:
        return True
    if string[0].isnumeric() or string[1].isnumeric() or string[2].isnumeric():
        return True

    # Get the first word in the string
    word = string.split(" ")[0]

    # If the word is greater than 20 characters,
    # it is almost 100% not a word
    if len(word) > 20 or len(string) == 0:
        return True

    # If the word has more numbers than
    # letters, it is not a word
    lets = "".join(re.findall("[a-zA-Z]+", word))
    nums = "".join(re.findall("[0-9]+", word))
    if len(nums) > len(lets)/2 and len(lets) > 1:
        return True




    # How many numbers are there?
    nums = "".join(re.findall("[0-9]+", word))
    if len(nums) > 5:
        return True

    
    # How many stupid characters does it have?
    stupid_chars = re.sub(r"[a-zA-Z0-9!.,?;:\-'\" ]+", "", string)
    if len(stupid_chars) > 4:
        return True




    # Try a second word
    try:
        word = string.split(" ")[1]
    except IndexError:
        return False
    
    # If the word is greater than 20 characters,
    # it is almost 100% not a word
    if len(word) > 20 or len(string) == 0:
        return True

    # If the word has more numbers than
    # letters, it is not a word
    lets = "".join(re.findall("[a-zA-Z]+", word))
    nums = "".join(re.findall("[0-9]+", word))
    if len(nums) > len(lets)/2 and len(lets) > 1:
        return True


    # Sentence is (probably) not trash
    return False


# Current file number
fileNum = 1
outFileDir = "data_clean"
outFile = open(outFileDir+os.sep+str(fileNum)+".txt", "w")

# Number of lines written to the file
numLines = 0

# Keep track of the previous string so its not repeated
prevStr = ""

# Iterate over all directories
for directory in os.scandir(dir_to_look_at):
    directory = directory.name

    # Look at all files in this directory
    for filename in os.scandir(dir_to_look_at + os.sep + directory):
        filename = filename.name

        # Get the file extension
        exten = filename.split(".")[-1]

        # Get the full filename path
        filename_full = dir_to_look_at+os.sep+directory+os.sep+filename

        # Skip the file if it has a bad extension
        if exten not in extens:
            continue

        # Open the file according to its extension
        if exten == "ass" or exten == "ssa" or exten == "srt":
            # Read in the file
            subtitles = parser.parse(filename_full)

            try:
                # Iterate over each data point
                for line in subtitles:
                    # Get the line data. Join by a space
                    line = " ".join(line.lines).strip()

                    # Skip the line if it is trash
                    if is_trash(line) == True:
                        continue

                    # Clean the line
                    line = "".join(re.findall(r"[a-zA-Z0-9!.,?;:\-'\" ]+", line))
                    
                    # Trash check 2
                    if is_trash(line) == True:
                        continue

                    # Replace double or more spaces
                    line = re.sub(r"[ ]{2,}", " ", line)

                    # If the line doesn't have a newline character,
                    # add one
                    if line[-1] != "\n":
                        line += "\n"

                    # Make sure the line isn't added multiple times
                    if line == prevStr:
                        continue

                    # Write the line to the output file
                    outFile.write(line)
                    numLines += 1

                    # If the line was written, make it the new
                    # previous string
                    prevStr = line

                    # If the number of lines written hit
                    # the threshold, open a new file
                    if numLines >= thresh:
                        # Close the current file
                        outFile.close()

                        # Increase the file count
                        fileNum += 1

                        # Open the new file
                        outFile = open(outFileDir+os.sep+str(fileNum)+".txt", "w")

                        # Number of lines is now 0
                        numLines = 0
                
            except UnicodeDecodeError:
                continue
            except pysubparser.classes.exceptions.InvalidTimestampError:
                continue 
            except:
                continue

# Close the file
outFile.close()