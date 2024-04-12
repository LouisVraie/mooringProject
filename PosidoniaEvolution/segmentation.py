import numpy as np

import matplotlib.pyplot as plt

import cv2
from PIL import Image, ImageEnhance

import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import sys

import supervision as sv

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

#lire l'image
# imageLue = Image.open("image_zones_claires_remplacees.jpg")
imageLue = Image.open("superposed_image.jpg")

#taille de l'image
width, height = imageLue.size
print(width)
#redimensionner
#imageComp=imageLue.resize((int(width/2.),int(height/2.)))

left = 100
top =0
right = width-100
bottom = height
 
img_res = imageLue.crop((left, top, right, bottom)) 

#sauvegarder l'image réduite
img_res.save("reduction_image_2022(t).jpeg")

img_to_segment = cv2.imread('reduction_image_2022(t).jpeg')
img_to_segment = cv2.cvtColor(img_to_segment, cv2.COLOR_BGR2RGB)
height, width = img_to_segment.shape[:2]
point_width_offset = width // 6

plt.figure(figsize=(10,10))
plt.imshow(img_to_segment)
plt.axis('on')
plt.show()

#selecting objects with sam
sys.path.append("..")

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

#put a pointer on the image to identifiate the posidonia
predictor.set_image(img_to_segment)
input_point = np.array([[point_width_offset,height/2],[width-point_width_offset,height/2]])
input_label = np.array([1,1])
plt.figure(figsize=(10,10))
plt.imshow(img_to_segment)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()  

#masks generation

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

#print the generated masks
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(img_to_segment)
    show_mask(mask, plt.gca())
    
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  

#print the masks
sv.plot_images_grid(
    images=masks,
    grid_size=(1, 4),
    size=(16, 4)
)

