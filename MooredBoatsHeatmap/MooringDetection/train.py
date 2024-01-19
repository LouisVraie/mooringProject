from ultralytics import YOLO
from PIL import Image
import torch
import cv2

CONFIG_FILE_PATH = "./config/configTA_V.yaml"

# Check if CUDA is available
worker = 0 if torch.cuda.is_available() else 8

# Load a model
model = YOLO("yolov8n.yaml").load('yolov8n.pt')  # build a new model from scratch
# model = YOLO("runs/detect/train26/weights/best.pt") # 500 epochs without background images
# model = YOLO("runs/detect/train44/weights/best.pt") # 100 + 300 epochs with background images

# Train the model
model.train(
    data=CONFIG_FILE_PATH,
    epochs=500,
    imgsz=640,
    batch=16,
    patience=100,
    workers=worker,
    # fraction=0.3,
    # dropout=0.2,
    optimizer='auto',
)

# Evaluate model performance on the validation set
metrics = model.val()

# Predict on an image
results = model.predict(
    "data/images/2018-08.png", 
    save=True, 
    imgsz=1632, 
    conf=0.5,
    line_width=1,
    show_conf=True,
    show_labels=False,
) 

# Plotting results
res_plotted = results[0].plot()
res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
img = Image.fromarray(res_plotted)

img.show()