from ultralytics import YOLO
from PIL import Image
import torch

# Check if CUDA is available
worker = 0 if torch.cuda.is_available() else 8

model = YOLO("runs/detect/train22/weights/best.pt") # load a pretrained model (recommended for training)

# results = model.predict("data/images/2018-08.png", save=True, imgsz=1632)  # predict on an image
results = model.predict(
  source="data/images/trainMovingBoat/boat2021.png", 
  save=True, 
  imgsz=640,
  # conf=0.3,
  line_width=1,
  # show_conf=True,
  # show_labels=False,
)  # predict on an image

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    print("boxes : ")
    print(boxes)
    masks = result.masks  # Masks object for segmentation masks outputs
    print("masks : ")
    print(masks)
    keypoints = result.keypoints  # Keypoints object for pose outputs
    print("keypoints : ")
    print(keypoints)
    probs = result.probs  # Probs object for classification outputs
    print("probs : ")
    print(probs)