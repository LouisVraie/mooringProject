from ultralytics import YOLO
from PIL import Image
import torch
import cv2

# Check if CUDA is available
worker = 0 if torch.cuda.is_available() else 8

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("runs/detect/train22/weights/best.pt") # load a pretrained model (recommended for training)
# model.to(device=device)

# Use the model
model.train(data="configMerge.yaml", epochs=100, imgsz=640, batch=32, patience=25, workers=worker)  # train the model

metrics = model.val()  # evaluate model performance on the validation set
results = model.predict("data/images/2018-08.png", save=True, imgsz=1632)  # predict on an image

# print(results)
# path = model.export(format="onnx") 

# Plotting results
res_plotted = results[0].plot()
res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
img = Image.fromarray(res_plotted)

img.show()

# img.save('runs/predicted/result.png')