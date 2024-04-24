from PIL import Image
import cv2 
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
IMG_DIR = os.path.join(SCRIPT_DIR, "images")
MASK_DIR = os.path.join(IMG_DIR, "mask")

def old_main():
    img1 = cv2.imread('./images/googleEarth/lerins2019-09-28.png')
    # img1 = cv2.imread('binair_image_2006.jpg')
    img2 = cv2.imread('binair_image_2022.jpeg')

    # Redimensionner les images pour qu'elles aient la même taille
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    # Calculer la différence entre les deux images
    diff = cv2.absdiff(img1, img2)
    print(diff[10,100])
    # Seuil pour la différence
    seuil = 30


    # Créer une image vide pour la superposition
    superposed_image = img1

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if diff[i, j][1] > seuil and img1[i, j][1] < 200:  # Changer 200 si besoin
                superposed_image[i, j] = [0, 0, 255]  # Rouge en BGR

    # Sauvegarder l'image superposée
    cv2.imwrite('image_superposee.jpg', superposed_image)

    # Afficher l'image superposée
    cv2.imshow('Image Superposée', superposed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def build_superposition_image():
    superposed_image = None
    
    # For each image in the mask directory
    for mask_file in os.listdir(MASK_DIR):
        
        mask_path = os.path.join(MASK_DIR, mask_file)
        mask = cv2.imread(mask_path)
        
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        # Create a new image with the mask
        if mask_file == os.listdir(MASK_DIR)[0]:
            superposed_image = mask
        else:
            # If the pixel of the superposed image is not black and the pixel of the mask is black the pixel is set to red
            superposed_image = np.where((superposed_image != [0, 0, 0]) & (mask == [0, 0, 0]), [255, 0, 0], superposed_image)
            
            # If the pixel of the superposed image is black and the pixel of the mask is not black the pixel is set to green
            superposed_image = np.where((superposed_image == [0, 0, 0]) & (mask != [0, 0, 0]), [0, 255, 0], superposed_image)
    
    if superposed_image is not None:
        # Convert the mask to 8-bit depth if it's not already
        superposed_image = cv2.convertScaleAbs(superposed_image)
        # Save the superposed image
        superposed_image = cv2.cvtColor(superposed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(IMG_DIR, "superposed_image.png"), superposed_image)
        
        # Remove the yellow pixels
        superposed_image = cv2.cvtColor(superposed_image, cv2.COLOR_BGR2RGB)    
        superposed_image[(superposed_image[:, :, 0] == 255) & (superposed_image[:, :, 1] == 255) & (superposed_image[:, :, 2] == 0)] = [255, 255, 255]
        
        # Save the superposed image without yellow pixels
        superposed_image = cv2.convertScaleAbs(superposed_image)
        superposed_image = cv2.cvtColor(superposed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(IMG_DIR, "superposed_image_no_yellow.png"), superposed_image)

if __name__ == '__main__':
    build_superposition_image()