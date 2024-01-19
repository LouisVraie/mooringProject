from PIL import Image
import numpy as np

def extraire_couleur(image_path, x, y):
    # Ouvrir l'image avec Pillow
    image = Image.open(image_path)

    # Obtenir la couleur du pixel à la position (x, y)
    couleur_pixel = image.getpixel((x, y))

    # Afficher la couleur extraite
    print("Couleur du pixel à la position ({}, {}): {}".format(x, y, couleur_pixel))

    return couleur_pixel

def remplacer_zones_claires_par_couleur(image_path, x=0, y=0, seuil_clair=20):
    # Ouvrir l'image avec Pillow
    image = Image.open(image_path)

    # Convertir l'image en tableau NumPy pour un accès facile aux pixels
    image_array = np.array(image)
    
    
    couleur_remplacement = extraire_couleur(image_path, x, y)
    # Appliquer la transformation des couleurs
    zones_claires_indices = (image_array[:, :, 0] > seuil_clair) & (image_array[:, :, 1] > seuil_clair) & (image_array[:, :, 2] > seuil_clair)
    image_array[zones_claires_indices] = couleur_remplacement

    # Convertir le tableau NumPy modifié en image Pillow
    image_resultat = Image.fromarray(image_array)

    # Afficher l'image originale et l'image résultante
    image.show(title="Image originale")
    image_resultat.show(title="Image avec les zones claires remplacées")

    # Sauvegarder l'image résultante
    image_resultat.save("image_zones_claires_remplacees.jpg")

image_path="image2022.jpeg"
image = Image.open(image_path)
width, height = image.size

remplacer_zones_claires_par_couleur(image_path,width/2,height-100)