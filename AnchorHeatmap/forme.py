import math
import cv2

image = cv2.imread('BaseImage/heatmapProofOfConceptcrop.jpg')
height, width, c = image.shape
matrix_width = round(width/10)
matrix_height = round(height/10)

print(width, height)


def set_pixel(matrix, x, y, value):
    # Ramener les indices aux bords si hors limites
    y = max(0, min(y, len(matrix) - 1))
    x = max(0, min(x, len(matrix[0]) - 1))
    
    # Modifier la valeur à l'emplacement spécifié
    matrix[y][x] = value

def calculate_line_coefficients(x1, y1, x2, y2):
    # Calculer la pente (m)
    m = (y2 - y1) / (x2 - x1)
    
    # Calculer l'ordonnée à l'origine (b)
    b = y1 - m * x1
    
    return m, b

def is_below_line(x, y, m, b):
    # Calculer la valeur attendue de y sur la droite
    expected_y = m * x + b
    # Vérifier si le point est en dessous de la droite
    return y < expected_y

def draw_line(matrix, x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    if dx > dy:
        err = dx / 2.0
        while x != x2:
            set_pixel(matrix, x, y, 1)
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            set_pixel(matrix, x, y, 1)
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    set_pixel(matrix, x, y, 1)

def indices_premier_dernier_un(ligne):
    if 1 not in ligne:
        return None, None  # Aucun 1 dans la ligne
    premier_un = ligne.index(1)
    dernier_un = len(ligne) - 1 - ligne[::-1].index(1)  
    return premier_un, dernier_un

def remplissage(matrice):
    for i in range (len(matrice)):
        a,b = indices_premier_dernier_un(matrice[i])
        if a != b:
            for j in range(a+1,b):
                matrice[i][j] = 1

def draw_triangle(matrix, x1, y1, x2, y2, x3, y3):
    
    draw_line(matrix, x1, y1, x2, y2)
    draw_line(matrix, x2, y2, x3, y3)
    draw_line(matrix, x3, y3, x1, y1)
    remplissage(matrix)
    
def draw_triangle_vide(matrix, x1, y1, x2, y2, x3, y3):
    
    draw_line(matrix, x1, y1, x2, y2)
    draw_line(matrix, x2, y2, x3, y3)
    draw_line(matrix, x3, y3, x1, y1)
    

def draw_circle(matrix, xc, yc, radius):
    xc,yc = yc,xc
    x = radius
    y = 0
    err = 0

    while x >= y:
        set_pixel(matrix, xc + x, yc + y, 1)
        set_pixel(matrix, xc + y, yc + x, 1)
        set_pixel(matrix, xc - y, yc + x, 1)
        set_pixel(matrix, xc - x, yc + y, 1)
        set_pixel(matrix, xc - x, yc - y, 1)
        set_pixel(matrix, xc - y, yc - x, 1)
        set_pixel(matrix, xc + y, yc - x, 1)
        set_pixel(matrix, xc + x, yc - y, 1)

        if err <= 0:
            y += 1
            err += 2 * y + 1
        if err > 0:
            x -= 1
            err -= 2 * x + 1
    remplissage(matrix)
    
def draw_circle_vide(matrix, xc, yc, radius):
    xc,yc = yc,xc
    x = radius
    y = 0
    err = 0

    while x >= y:
        set_pixel(matrix, xc + x, yc + y, 1)
        set_pixel(matrix, xc + y, yc + x, 1)
        set_pixel(matrix, xc - y, yc + x, 1)
        set_pixel(matrix, xc - x, yc + y, 1)
        set_pixel(matrix, xc - x, yc - y, 1)
        set_pixel(matrix, xc - y, yc - x, 1)
        set_pixel(matrix, xc + y, yc - x, 1)
        set_pixel(matrix, xc + x, yc - y, 1)

        if err <= 0:
            y += 1
            err += 2 * y + 1
        if err > 0:
            x -= 1
            err -= 2 * x + 1
    

def addition_matrice(matrix, tmp):
    for j in range(matrix_height):
        for k in range(matrix_width):
            matrix[j][k] += tmp[j][k]

def soustraction_matrice(matrix, tmp):
    for j in range(matrix_height):
        for k in range(matrix_width):
            matrix[j][k] -= tmp[j][k]

#fonction qui realise la conversion d'une coordonnee en degre, minute, seconde en degre decimaux
def coord(degree, minutes, secondes):
    return degree + minutes/60 + secondes/3600

'''HAUT GAUCHE : 43°31'3.18"N 7°2'3.76"E
BAS DROITE : 43°30'28.09"N 7°3'47.92"E'''

HautGauche = (coord(43, 31, 3.18), coord(7, 2, 3.76))
BasGauche = (coord(43, 30, 28.09), coord(7, 2, 3.76))
BasDroite = (coord(43, 30, 28.09), coord(7, 3, 47.92))
HautDroite = (coord(43, 31, 3.18), coord(7, 3, 47.92))

pixelNorth = abs(BasDroite[0] - HautGauche[0])/height
pixelEast = abs(BasDroite[1] - HautGauche[1])/width

def coord2pixel(coord):
    x = int(abs(HautGauche[1] - coord[1])/pixelEast/10)
    y = int(abs(HautGauche[0] - coord[0])/pixelNorth/10)
    return (x, y)

def haversine_distance(lat1, lon1, lat2, lon2):
    # Rayon moyen de la Terre en mètres
    earth_radius = 6371000.0
    
    # Conversion des coordonnées de degrés à radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Calcul des écarts
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Formule de la haversine
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Distance en mètres
    distance = earth_radius * c
    
    return distance

distanceNord = haversine_distance(HautGauche[0], HautGauche[1], BasGauche[0], BasGauche[1])
distanceEst = haversine_distance(HautGauche[0], HautGauche[1], HautDroite[0], HautDroite[1])

distancepixelNord = distanceNord/height
distancepixelEst = distanceEst/width

#print(distancepixelNord, distancepixelEst)
distancepixelmoyenne = (distancepixelNord + distancepixelEst)/2
#print(distancepixelmoyenne)

print("Distance Nord-Sud:", distanceNord,"m Distance Est-Ouest:", distanceEst, "m")
print("Distance pixel Nord-Sud:", distancepixelNord, "m Distance pixel Est-Ouest:", distancepixelEst, "m")
print("Distance pixel moyenne:",distancepixelmoyenne,"m")

import numpy as np

def find_base_points(x0, y0, height, orientation_angle):
    # Convert the angle in rad
    orientation_angle_rad = math.radians(orientation_angle)

    # Calculate the horizontal distances between the vertex and each points of the base
    dy = (1 / math.sqrt(3)) * height
    
    # Calculate the coord of the points of the base before rotation
    x1_unrotated = x0 - dy / 2
    y1_unrotated = y0 - height
    
    x2_unrotated = x0 + dy / 2
    y2_unrotated = y0 - height
    
    # Rotation matrix for the orientation angle
    rotation_matrix = np.array([[math.cos(orientation_angle_rad), -math.sin(orientation_angle_rad)],
                                [math.sin(orientation_angle_rad), math.cos(orientation_angle_rad)]])
    
    # Apply the rotation to the coord of the points of the base
    rotated_point1 = np.dot(rotation_matrix, np.array([x1_unrotated - x0, y1_unrotated - y0])) + np.array([x0, y0])
    rotated_point2 = np.dot(rotation_matrix, np.array([x2_unrotated - x0, y2_unrotated - y0])) + np.array([x0, y0])
    
    return round(rotated_point1[0]), round(rotated_point1[1]), round(rotated_point2[0]), round(rotated_point2[1])
