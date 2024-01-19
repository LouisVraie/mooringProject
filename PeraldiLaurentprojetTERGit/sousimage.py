import os
import pandas as pd
import orientation as ori
from PIL import Image

def creer_sous_image(image_path, coord_x1, coord_y1, coord_x2, coord_y2, id,orientation):
    # Charger l'image
    image = Image.open(image_path)

    # Créer une sous-image à partir des coordonnées fournies
    sous_image = image.crop((coord_x1, coord_y1, coord_x2, coord_y2))
    # Enregistrer la sous-image
    sous_image_oriente = sous_image.rotate(orientation, expand=True)
    sous_image_oriente.save("Orientation\\input\\sous_image"+str(id)+".png")

# Exemple d'utilisation

def trouver_angle(id):
    # load image as HSV and select saturation
    repo = "Orientation\\input\\sous_image"+str(id)+".png"
    return ori.trouver_angle(repo)

def sousimagemain(data_path):
    result = []
    data = pd.read_csv(data_path)
    id=0
    for index, row in data.iterrows():
        image_path = row['image']
        orientation = row['orientation']
        haut, gauche, bas, droite = int(row['boxe_top_left_x']), int(row['boxe_top_left_y']), int(row['boxe_bottom_right_x']), int(row['boxe_bottom_right_y'])
        id +=1
        print(id)
        creer_sous_image(image_path,haut,gauche,bas,droite,id,orientation)
        angle = round(trouver_angle(id))
        coordonnees = row['lat'], row['lng']
        result.append([coordonnees, angle])
    return result
