import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.join(SCRIPT_DIR, "..","data")

FOLDER = "augmentTrain"
IMAGES_DIR = os.path.join(BASE_DIR, "images", FOLDER)
LABELS_DIR = os.path.join(BASE_DIR, "labels", FOLDER)

images_list = [os.path.splitext(path.name)[0] for path in os.scandir(IMAGES_DIR) if path.is_file()]
labels_list = [os.path.splitext(path.name)[0] for path in os.scandir(LABELS_DIR) if path.is_file()]

end_images_list = images_list.copy()
end_labels_list = labels_list.copy()

for file in images_list:
  if file in labels_list:
    end_images_list.remove(file)
    end_labels_list.remove(file)
    
print("Missing images related to a label :")
print(end_images_list)
print(end_labels_list)

images_to_delete = [file+".png" for file in end_images_list]
labels_to_delete = [file+".txt" for file in end_labels_list]

# Removing files
def removeFiles(dir_path, file_list):
  for file in file_list:
    filepath = os.path.join(dir_path, file)
    if os.path.isfile(filepath):
      os.remove(filepath)
      
removeFiles(IMAGES_DIR, images_to_delete)
removeFiles(LABELS_DIR, labels_to_delete)