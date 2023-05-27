import urllib.request
from bs4 import BeautifulSoup as BS
import urllib.request
import os
import zipfile
import shutil
import py7zr
import patoolib

# Read in the page
html = urllib.request.urlopen("https://www.kitsunekko.net/dirlist.php?dir=subtitles%2F").read()
soup = BS(html, 'html.parser')

restart = 278


# Get all table elements
elems = soup.find_all("tr")

# Count up the number of directories to make a nice file structure
dirs = restart

# List of all file endings found
endings = set()

# Iterate over all elements to get the info
for element in elems[restart:]:
    # Get the link
    link = "https://www.kitsunekko.net" + element.find_all("a")[0].attrs["href"]

    # Open the page
    html2 = urllib.request.urlopen(link).read()
    soup2 = BS(html2, 'html.parser')

    # Get all table links
    elems2 = soup2.find_all("tr")

    # New directory
    dirs += 1
    if not os.path.exists(os.getcwd() + os.sep + str(dirs)):
        os.mkdir(os.getcwd() + os.sep + str(dirs))

    # Count the number of files
    files = 0

    # Iterate over each link
    for element2 in elems2:
        # Get the download link
        download_link = "https://www.kitsunekko.net/" + element2.find_all("a")[0].attrs["href"]
        download_link = download_link.replace(" ", "%20").replace("⁄", "/")

        # Get the file ending
        ending = download_link.split(".")[-1]
        if len(download_link.split(".")) > 1:
            if download_link.split(".")[-2] == "ass" or  download_link.split(".")[-2] == "srt" or  download_link.split(".")[-2] == "ssa":
                ending = download_link.split(".")[-2] + "." + download_link.split(".")[-1]
        endings.add(ending)
        
        # Download the file
        file_path = f"{os.getcwd()}{os.sep}{dirs}"
        file_name = f"{file_path}{os.sep}{files}.{ending}"
        file_name_no_end = f"{file_path}{os.sep}{files}"
        try:
            urllib.request.urlretrieve(download_link, file_name)
        except urllib.request.HTTPError:
            continue
        except UnicodeEncodeError:
            continue
        except OSError:
            continue

        # If the ending is a zip, unzip it here
        try:
            if ending == "zip":
                with zipfile.ZipFile(file_name, 'r') as zip_ref:
                    zip_ref.extractall(file_name_no_end)
            
            if ending == "7z":
                with py7zr.SevenZipFile(file_name, 'r') as zip_ref:
                    zip_ref.extractall(file_name_no_end)

            if ending == "rar":
                if not os.path.exists(file_name_no_end):
                    os.mkdir(file_name_no_end)
                patoolib.extract_archive(file_name, outdir=file_name_no_end, program="7z", verbosity=-1, interactive=False)
        except:
            continue

        # Copy files from in the zip folder
        if ending == "zip" or ending == "7z" or ending =="rar":
            # Recursive copying
            files -= 1
            for root, dirs_, files_ in os.walk(f"{file_path}{os.sep}{files+1}"):
                for file in files_:
                    file = root + os.sep + file
                    # # Copy the contents in the zip to this dir
                    # files -= 1
                    # for file in os.listdir(f"{file_path}{os.sep}{files+1}"):
                    #     files += 1
                    #     ending = file.split(".")[-1]
                    #     endings.add(ending)
                    #     shutil.move(file_name_no_end + os.sep + file, f"{file_path}{os.sep}{files}.{ending}")

                    # # Copy the contents in the zip to this dir
                    # files += 1
                    # ending = file.split(".")[-1]
                    # endings.add(ending)
                    # shutil.move(file_name_no_end + os.sep + file, f"{file_path}{os.sep}{files}.{ending}")

                    # Copy the contents in the zip to this dir
                    files += 1
                    ending = file.split(".")[-1]
                    endings.add(ending)
                    try:
                        shutil.move(file, f"{file_path}{os.sep}{files}.{ending}")
                    except FileNotFoundError:
                        pass

            # Delete the zip file and empty directory
            if len(file_name_no_end) < 35:
                continue
            try:
                shutil.rmtree(file_name_no_end)
                os.remove(file_name)
            except FileNotFoundError:
                continue

        files += 1

    print(f"Loaded: {dirs}    Endings: {endings}")