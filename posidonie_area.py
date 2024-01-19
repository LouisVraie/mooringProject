import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
'''
img =cv2.imread('image2022.jpeg')

# convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding in the gray image to create a binary image
ret,thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)

# Find the contours using binary image
contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
print("Number of contours in image:",len(contours))
cnt = contours[0]

# compute the area and perimeter
area = cv2.contourArea(cnt)

print('Area:', area)
img1 = cv2.drawContours(img, [cnt], -1, (0,0,255), 100)
x1, y1 = cnt[0,0]
print(x1,y1)
cv2.putText(img1, f'Area:{area}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 200)
'''




def calcul_taille_pixel(hauteur, angle_vue,nb_pixels_largeur):
    
    taille_reelle = 2* hauteur / math.tan(angle_vue/2) 
    taille_reelle=taille_reelle / nb_pixels_largeur
    return taille_reelle

def plot_historique(areas):
    x=[]
    y=[]
    for key, value in areas.items():
        x.append(key)
        y.append(value)
    
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.xlabel("year")
    plt.ylabel("area (m^2)")
    plt.title("evolution of Posidonie area")
    plt.show() 
    plt.savefig("evolution_of_Posidonie_area")
         
photos=5 
areas_pixel={"2004": 15200, "2006": 14800,"2010":14300,"2014":13800,"2019":13000,"2022":12600,"2023":12500}
areas=np.empty(7)

angle_vue=77
hauteur=1238 
largeur=1455
i=0
taille_reelle_pixel = calcul_taille_pixel(hauteur, angle_vue,largeur)
for key,value in areas_pixel.items() :
    value*=taille_reelle_pixel 
    areas[i]=value
    i+=1


print("Taille réelle d'un pixel :"+"{:.2f}".format(taille_reelle_pixel)  + "metre")

#simulation 
import matplotlib.pyplot as plt
import numpy as np
keys = list(areas_pixel.keys())
values = list(areas_pixel.values())

# Tracer le graphe
plt.bar(keys, values)
plt.title("Évolution de la surface de la posidonie")
plt.xlabel("Année")
plt.ylabel("Surface de la posidonie (m^2)")
plt.legend()
plt.grid(True)
plt.show()

# Données des 10 dernières années (remplacez ces valeurs par les vôtres)
annees_passes = np.array([2004,2006,2010,2014,2019, 2022, 2023])
# Ajustement d'un polynôme de degré 2 (vous pouvez ajuster le degré en fonction de vos données)
coefficients = np.polyfit(annees_passes, areas, 2)
polynome = np.poly1d(coefficients)

# Extrapolation pour les 10 prochaines années
annees_futures = np.arange(2023, 2033)
surface_posidonie_futures = polynome(annees_futures)

# Représentation graphique
plt.plot(annees_passes, areas, marker='o', label='Données passées')
plt.plot(annees_futures, surface_posidonie_futures, marker='o', linestyle='dashed', label='Extrapolation')
plt.title("Évolution de la surface de la posidonie")
plt.xlabel("Année")
plt.ylabel("Surface de la posidonie (unité)")
plt.legend()
plt.grid(True)
plt.show()
