from PIL import Image
import cv2 
import numpy as np
img1 = cv2.imread('lerins2006.jpeg')
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