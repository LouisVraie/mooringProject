import cv2
import numpy as np

def histogram_matching(image, reference):
    # Convertir les images en niveaux de gris
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    # Calculer les histogrammes des deux images
    hist_image = cv2.calcHist([image_gray], [0], None, [256], [0,256])
    hist_reference = cv2.calcHist([reference_gray], [0], None, [256], [0,256])

    # Calculer les fonctions de distribution cumulée (CDF) des histogrammes
    cdf_image = hist_image.cumsum()
    cdf_reference = hist_reference.cumsum()

    # Normaliser les CDF pour obtenir des valeurs entre 0 et 1
    cdf_image_normalized = cdf_image / cdf_image[-1]
    cdf_reference_normalized = cdf_reference / cdf_reference[-1]

    # Appliquer la correspondance des histogrammes
    mapping = np.interp(cdf_image_normalized, cdf_reference_normalized, range(256))
    matched_image = cv2.LUT(image_gray, mapping.astype('uint8'))

    # Convertir l'image égalisée en couleur
    matched_image_color = cv2.cvtColor(matched_image, cv2.COLOR_GRAY2BGR)

    return matched_image_color

# Charger l'image à égaliser et l'image de référence
name = 'lerins2019'
image_to_match = cv2.imread(name + ".jpeg")
reference_image = cv2.imread('image2022.jpeg')

# Appliquer l'égalisation de spécification d'histogramme
matched_image = histogram_matching(image_to_match, reference_image)

# Sauvegarder l'image égalisée
cv2.imwrite(f'./images/matched_{name}.jpeg', matched_image)

# Afficher les images
# cv2.imshow('Original Image', image_to_match)
# cv2.imshow('Reference Image', reference_image)
# cv2.imshow('Matched Image', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()