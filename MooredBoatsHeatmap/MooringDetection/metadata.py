import subprocess
import json
import os

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

EXIFTOOL_PATH = os.path.join(SCRIPT_PATH, "exiftool")

def extract_metadata(image_path: str) -> dict:
    """Extract metadata from an image and save it in a json file.

    Args:
        image_path (str): Path to the image.
    Returns:
        dict: Metadata dictionnary of the given image.
    """
    exiftool_output = subprocess.check_output([EXIFTOOL_PATH, image_path, "-j"])
    meta_data = json.loads(exiftool_output.decode('utf-8'))
    
    return meta_data[0]