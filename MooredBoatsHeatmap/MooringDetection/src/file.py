import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Source folder (where the files to copy are located)
SOURCE_DIR = os.path.join(SCRIPT_DIR, "../data/images/augmentVal")

# Destination folder (where the .txt files will be created)
DESTINATION_DIR = os.path.join(SCRIPT_DIR, "../data/labels/augmentVal")

# Check if the source folder exists
if not os.path.exists(SOURCE_DIR):
    print(f"The source folder {SOURCE_DIR} does not exist !!!")
# Check if the destination folder exists
if not os.path.exists(DESTINATION_DIR):
    print(f"The destination folder {DESTINATION_DIR} does not exist !!!")

count = 0
# For each file in the source folder
for fichier in os.listdir(SOURCE_DIR):
    # if the file name ends with ".png"
    if fichier.endswith(".png"):
        updateFile = fichier.replace(".png", ".txt")
    # if the file name ends with ".jpg"
    if fichier.endswith(".jpg"):
        updateFile = fichier.replace(".jpg", ".txt")
    
    # Check if the file is a file (not a folder)
    if os.path.isfile(os.path.join(SOURCE_DIR, fichier)):
        # Check if the file exists in the 
        if not os.path.isfile(os.path.join(DESTINATION_DIR, updateFile)):
            # Create the full path of the .txt file in the destination folder
            chemin_destination = os.path.join(DESTINATION_DIR, updateFile)
        
            # Create a new .txt file in the destination folder
            open(chemin_destination, 'w').close()
            count += 1

print(count, "Files .txt created successfully in the destination folder.")