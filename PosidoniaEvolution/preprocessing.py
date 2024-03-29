from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_main_color(image: np.ndarray) -> tuple:

    # Calculate the RGB histogram
    rgb_histogram, rgb_bins = np.histogramdd(image.reshape(-1, 3), bins=256, range=[(0, 256), (0, 256), (0, 256)])

    # Flatten the 3D RGB histogram into a 1D array
    rgb_histogram = np.array([val for sublist in rgb_histogram for val in sublist])

    # Get the most common RGB color (the index with the highest value)
    most_common_color_index = np.argmax(rgb_histogram)

    # Get the RGB values of the most common color
    r, g, b = most_common_color_index // (256 ** 2), most_common_color_index // 256 % 256, most_common_color_index % 256

    return r, g, b

def display_rgb_color(rgb_color: tuple):
    """
    Displays the given RGB color.
    
    :param rgb: A list or tuple of 3 float values between 0 and 1 representing the RGB color.
    """
    # Normalize the RGB color values if they're not within the 0-1 range
    if not all(0.0 <= c <= 1.0 for c in rgb_color):
        rgb_color = tuple(min(max(c / 255, 0.0), 1.0) for c in rgb_color)
    
    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # Create a rectangle with the given color and set the size
    rect = plt.Rectangle((0, 0), 1, 1, facecolor=rgb_color)

    # Add the rectangle to the axes
    ax.add_patch(rect)

    # Set the axes limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Hide the x and y axis labels
    plt.gca().set_axis_off()

    # Display the figure
    plt.show()

def extraire_couleur(image_path, x, y):
    # Ouvrir l'image avec Pillow
    image = Image.open(image_path)

    # Obtenir la couleur du pixel à la position (x, y)
    couleur_pixel = image.getpixel((x, y))

    # Afficher la couleur extraite
    print("Couleur du pixel à la position ({}, {}): {}".format(x, y, couleur_pixel))

    return couleur_pixel

def apply_blue_mask(image: np.ndarray, blue_lower: list[int] = [80, 40, 40], blue_upper: list[int] = [130, 255, 255]):
    
    # Convert the image to HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for blue
    blue_lower = np.array(blue_lower, dtype=np.uint8)
    blue_upper = np.array(blue_upper, dtype=np.uint8)

    # Apply the blue mask to the image
    blue_mask_hsv = cv2.inRange(image_hsv, blue_lower, blue_upper)
    
    # Remove the noise
    blue_mask_hsv = cv2.medianBlur(blue_mask_hsv, 13)

    # Apply the blue mask to the original image
    image_blue = cv2.bitwise_and(image, image, mask=blue_mask_hsv)

    # Display the original image and the blue-filtered image
    # cv2.imshow('Blue-Filtered Image', image_blue)
    
    # Return the blue-filtered image in normal RGB color space
    return image_blue

def get_main_color_around_pixel(image, default_color, x, y, radius):
    # Extract the frame around the pixel
    frame = image[max(0, y - radius):min(image.shape[0], y + radius),
                  max(0, x - radius):min(image.shape[1], x + radius)]
    
    # Convert the frame into a one-dimensional list of RGB pixels
    pixels = frame.reshape(-1, 3)
    
    # Count occurrences of each color in the frame
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    
    if len(counts) == 0:
        # No unique colors found in the region, return a default color
        return default_color
    elif len(counts) == 1:
        # Only one unique color found in the region, return that color if its not black
        if tuple(unique_colors[0]) != (0, 0, 0):
            return unique_colors[0]
        else:
            return default_color
    else:
        # Find the main color (excluding the color of the central pixel)
        max_count_idx = np.argmax(counts)
        main_color = unique_colors[max_count_idx]
        second_max_count_idx = np.argsort(counts)[-2]
        second_max_color = unique_colors[second_max_count_idx]
        
        # If the two most common colors have the same count, return a default color
        if counts[max_count_idx] == counts[second_max_count_idx]:
            if tuple(main_color) == default_color:
                return second_max_color
            else:
                return main_color

    # Find the main color (excluding the color of the central pixel and the default color) 
    main_color = unique_colors[np.argmax(counts[counts != np.max(counts)])]
    
    return main_color
    
def replace_main_color(image: np.ndarray, main_color: tuple[int, int, int], radius: int = 5):
    # result = image.copy()
    # Replace the main color with the most common color around each pixel without taking the main color into account
    for y in range(radius, image.shape[0] - radius):
        for x in range(radius, image.shape[1] - radius):
            if np.all(image[y, x] == main_color):
                image[y, x] = get_main_color_around_pixel(image, main_color, x, y, radius)
                
    return image
                
def remplacer_zones_claires_par_couleur(image: np.ndarray, x=0, y=0, seuil_clair=20, seuil_sombre=45):

    # main_color = extraire_couleur(image_path, x, y)
    main_color = get_main_color(image)
    print(f'The main color of the image is: {main_color}')
    
    # Apply the blue mask to the image
    blue_filtered_image = apply_blue_mask(image)
    
    # Récupérer les indices des pixels clairs
    zones_claires_indices = (image[:, :, 0] > seuil_clair) & (image[:, :, 1] > seuil_clair) & (image[:, :, 2] > seuil_clair)
    image[zones_claires_indices] = main_color
    
    # Get indexes of dark pixels
    # dark_indices = (image[:, :, 0] < seuil_sombre) & (image[:, :, 1] < seuil_sombre) & (image[:, :, 2] < seuil_sombre)
    # image[dark_indices] = main_color
    
    # If the pixel is bright and black in blue_filtered_image replace it with the black color
    black_color = [0, 0, 0]
    bright_black_indices = (blue_filtered_image[:, :, 0] == 0) & (blue_filtered_image[:, :, 1] == 0) & (blue_filtered_image[:, :, 2] == 0)
    image[bright_black_indices] = black_color
    
    # Thresholding to segment shadows

    cv2.imshow('Image avec les zones claires remplacées 1', image)
    # cv2.waitKey(0)
    # exit()
    
    # Replace the main color with the most common color around each pixel without taking the main color into account
    replaced_image = replace_main_color(image, main_color, radius=20)
    
    cv2.imshow('Image avec les zones claires remplacées 2', replaced_image)

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Binary', binary)
    
    # Remove the noise
    replaced_image = cv2.medianBlur(replaced_image, 3)
    
    # Adjust the contrast and brightness
    alpha = 1.2  # Contrast factor
    beta = 5    # Brightness factor
    image_array = cv2.convertScaleAbs(replaced_image, alpha=alpha, beta=beta)
    
    # Combine original and modified images by fusion (superposition)
    # image_array = cv2.addWeighted(original_image_array, 0.2, image_array, 0.8, 0)

    # Convertir le tableau NumPy modifié en image OpenCV
    image_resultat = image_array

    # Afficher l'image originale et l'image résultante
    cv2.imshow("Image avec les zones claires remplacées", image_resultat)

    # Attendre une touche pour fermer la fenêtre
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Sauvegarder l'image résultante
    cv2.imwrite("image_zones_claires_remplacees.jpg", image_resultat)
    
    return image_resultat


if __name__ == "__main__":
    # Charger l'image originale
    original_image=cv2.imread("image2022.jpeg")
    # original_image=cv2.imread("lerins2018.jpeg")
    
    # Change image contrast and brightness
    # alpha = 1.1 # Contrast factor
    # beta = 3    # Brightness factor
    # original_image = cv2.convertScaleAbs(original_image, alpha=alpha, beta=beta)
    
    # main_color = get_main_color(original_image)
    # print(f'The main color of the image is: {main_color}')
    # cv2.imshow('Contrast and Brightness Adjusted Image', original_image)
    # cv2.waitKey(0)
    # exit()
    width, height = original_image.shape[:2]

    remplacer_zones_claires_par_couleur(original_image, width/2, height-100, 50)