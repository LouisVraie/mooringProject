from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

from contours import apply_edges


def get_main_color(image: np.ndarray, is_grayscale: bool=False) -> tuple:

    if not is_grayscale:
        # Calculate the RGB histogram
        rgb_histogram, rgb_bins = np.histogramdd(image.reshape(-1, 3), bins=256, range=[(0, 256), (0, 256), (0, 256)])

        # Flatten the 3D RGB histogram into a 1D array
        rgb_histogram = np.array([val for sublist in rgb_histogram for val in sublist])

        # Get the most common RGB color (the index with the highest value)
        most_common_color_index = np.argmax(rgb_histogram)

        # Get the RGB values of the most common color
        r, g, b = most_common_color_index // (256 ** 2), most_common_color_index // 256 % 256, most_common_color_index % 256

        return r, g, b
    else:
        # Calculate the grayscale histogram
        grayscale_histogram, grayscale_bins = np.histogram(image, bins=256, range=(0, 256))

        # Get the most common grayscale color (the index with the highest value)
        most_common_color_index = np.argmax(grayscale_histogram)

        # Get the grayscale value of the most common color
        return most_common_color_index

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
    # cv2.waitKey(0)
    # exit()
    # Return the blue-filtered image in normal RGB color space
    return image_blue

def get_main_color_around_pixel(image, default_color, x, y, radius, is_grayscale=False):
    """
    Get the main color around a pixel within a given radius.

    Args:
        image (numpy.ndarray): The input image.
        default_color (int or tuple): The default color. If the function fails to detect any distinct color,
            this default color will be returned.
        x (int): The x-coordinate of the pixel.
        y (int): The y-coordinate of the pixel.
        radius (int): The radius around the pixel to consider.
        is_grayscale (bool, optional): Whether the image is grayscale. Defaults to False.

    Returns:
        int or tuple: The main color around the pixel. This could be a grayscale intensity value if the image is grayscale,
            or an RGB tuple if the image is in color.
    """
    # Get the height and width of the image based on whether it's grayscale or color
    if is_grayscale:
        h, w = image.shape
    else:
        h, w, _ = image.shape

    # Define the boundaries of the frame around the pixel
    ymin = max(0, y - radius)
    ymax = min(h, y + radius + 1)
    xmin = max(0, x - radius)
    xmax = min(w, x + radius + 1)

    # Extract the frame around the pixel
    frame = image[ymin:ymax, xmin:xmax]

    # Get unique colors and their counts in the frame
    unique_colors, counts = np.unique(frame, return_counts=True)

    # If no unique colors are found, return the default color
    if len(unique_colors) == 0:
        return default_color
    # If only one unique color is found or two unique colors are found with one being black,
    # return the default color if it's black, otherwise return the unique color
    elif len(unique_colors) == 1 or (len(unique_colors) == 2 and unique_colors[0] == 0):
        return default_color if unique_colors[0] == 0 else unique_colors[0]
    else:
        # Find the color with the maximum count
        max_count_idx = np.argmax(counts)
        max_color = unique_colors[max_count_idx]
        max_count = counts[max_count_idx]

        # Find the color with the second maximum count
        second_max_count = 0
        second_max_color = None
        for i, count in enumerate(counts):
            # Ensure proper comparison, especially when dealing with RGB colors
            if count > second_max_count and not np.array_equal(unique_colors[i], default_color) and not np.array_equal(unique_colors[i], max_color):
                second_max_count = count
                second_max_color = unique_colors[i]

        # If the second maximum count is equal to the maximum count, return the second maximum color,
        # otherwise return the main color
        return second_max_color if max_count == second_max_count else max_color

    
def replace_main_color(image: np.ndarray, main_color: tuple[int, int, int], radius: int = 5, is_grayscale: bool = False):
    
    if is_grayscale:
        h, w = image.shape
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        main_color = (main_color[0],) * 3  # Convert main_color to grayscale tuple for comparison
    else:
        h, w, _ = image.shape
    main_color = np.array(main_color)  # Convert main_color to numpy array for comparison
    
    # Generate meshgrid of coordinates for each pixel in the image
    y_coords, x_coords = np.ogrid[radius:h-radius, radius:w-radius]
    
    # Find indices where image equals main_color
    main_color_indices = np.where(np.all(image[y_coords, x_coords] == main_color, axis=-1))

    # Extract x and y indices
    y_indices, x_indices = main_color_indices
    
    # Replace main color with most common color around each pixel
    for y, x in zip(y_indices, x_indices):
        image[y, x] = get_main_color_around_pixel(image, main_color, x, y, radius)
    
    return image.squeeze() if is_grayscale else image


def replace_bright_pixels(image: np.ndarray, threshold: int = 80):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get the main color of the image
    main_color = get_main_color(gray, is_grayscale=True)
    print(f'The main color of the image is: {main_color}')
    
    # Get the indices of bright pixels
    bright_indices = gray > threshold
    
    # Replace the bright pixels with the main color
    gray[bright_indices] = main_color

    # Replace the main color with the most common color around each pixel without taking the main color into account
    replaced_image = replace_main_color(gray, (main_color, main_color, main_color), radius=5, is_grayscale=True)
    
    # Noise factor must be odd and greater than 1 and take into account image width 
    noise_factor = 3
    image_width_factor = gray.shape[1] // 1000
    noise_factor = noise_factor + image_width_factor 
    if noise_factor % 2 == 0:
        noise_factor += 1
    print(f'Noise factor: {noise_factor}')
    
    # Remove the noise
    replaced_image = cv2.medianBlur(replaced_image, noise_factor)
    # Return the image with the bright pixels replaced in grayscale
    return replaced_image 
    
    
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

    # cv2.imshow('Image avec les zones claires remplacées 1', image)
    # cv2.waitKey(0)
    # exit()
    
    # Replace the main color with the most common color around each pixel without taking the main color into account
    replaced_image = replace_main_color(image, main_color, radius=20)
    
    # cv2.imshow('Image avec les zones claires remplacées 2', replaced_image)

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Binary', binary)
    
    # Remove the noise
    replaced_image = cv2.medianBlur(replaced_image, 3)
    
    # Adjust the contrast and brightness on other pixels than black
    alpha = 1.8  # Contrast factor
    beta = 15    # Brightness factor
    image_array = cv2.convertScaleAbs(replaced_image, alpha=alpha, beta=beta)
    
    # Combine original and modified images by fusion (superposition)
    # image_array = cv2.addWeighted(original_image_array, 0.2, image_array, 0.8, 0)

    # Convertir le tableau NumPy modifié en image OpenCV
    image_resultat = image_array

    # Afficher l'image originale et l'image résultante
    # cv2.imshow("Image avec les zones claires remplacées", image_resultat)

    # Attendre une touche pour fermer la fenêtre
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Sauvegarder l'image résultante
    cv2.imwrite("image_zones_claires_remplacees.jpg", image_resultat)
    
    return image_resultat


if __name__ == "__main__":
    start_time = time.time()
    # Charger l'image originale
    # original_image=cv2.imread("image2022.jpeg")
    original_image=cv2.imread("./images/googleEarth/lerins2019-09.png")
    is_grayscale = True
    
    if is_grayscale:
        replaced_image = replace_bright_pixels(original_image)
        # Display the image with the bright pixels replaced
        # plt.figure(figsize=(12, 6))
        # plt.imshow(replaced_image, cmap='gray')
        # plt.show()
        
        apply_edges(replaced_image, boat_threshold=50, edges_low_threshold=36, edges_high_threshold=38, edges_presence=0.5, is_grayscale=True)
    
    else:
        # Color
        width, height = original_image.shape[:2]

        replaced_image = remplacer_zones_claires_par_couleur(original_image, width/2, height-100, 50)
        
        apply_edges(replaced_image)
        
    end_time = time.time()
    print(f"Execution time: {(end_time - start_time):.4f} seconds")
        
    