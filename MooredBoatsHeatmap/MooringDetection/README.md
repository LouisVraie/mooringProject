# Mooring project
---

Here is the boat detection part that compute GPS coordinates of moored boats on an image and putting the results in `predictResults.csv` for all detections and `noOverlaps_predictResults.csv` for detections without overlaps between images.

## Use
To generate the csv files run `predict.py` by given at the begining of the file the path of the folder to predict.

## Main files / folders of the project

- `config/` : contains configurations files for YOLOv8 training
- `data/` : all the data containing images and labels in the yolo format (download the original ones from [OneDrive](https://unice-my.sharepoint.com/:u:/g/personal/louis_vraie_etu_unice_fr/EWeSaTfSWQ1Os4SxNwb_YYoBCKG8EHOVWwE-zzapWBYG8g?e=lGeGVl))
  - `data/images/` : folder containing folders with train or validation images
  - `data/labels/` : folder containing folders with train or validation labels of the related images of `data/images/`
- `model/` : contains the best model
- `runs/` : (not present by default in the repo) generated folder by YOLOv8 that contains results of trainings, validations, predictions, ...
- `src/` : folder with the useful functions to run different things
- `yolo_data_augmentations` : [Fork](https://github.com/muhammad-faizan-122/yolo-data-augmentation) of a code that makes data augmentation, used in `augment.py`
- `augment.py` : apply data augmentation on a specific folder the output is generated in `yolo_data_augmentations/out-aug-ds/`
- `metadata.py` : extract the metadata of a given image using `exiftool`
- `train.py` : train the YOLOv8 model
- `predict.py` : detect boats of the given folder and create `predictResults.csv` and `noOverlaps_predictResults.csv` files


## Credits
Originally made by **[Louis Vraie](https://github.com/LouisVraie/)** for a multidisciplinary research project at **Côte d'Azur University, France**