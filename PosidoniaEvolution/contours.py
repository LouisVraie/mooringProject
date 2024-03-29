import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(image, title='Image', cmap_type='gray'):
  plt.figure(figsize=(10, 5))
  build_image(image, title, cmap_type)
  plt.show()

def show_two_images(image1, title1, image2, title2, cmap_type1='gray', cmap_type2='gray'):
  plt.figure(figsize=(14, 6))
  plt.subplot(2, 1, 1)
  build_image(image1, title1, cmap_type1)

  plt.subplot(2, 1, 2)
  build_image(image2, title2, cmap_type2)
  
  plt.show()

# Display the image with edges
def build_image(image, title='Edges', cmap_type=None):
  plt.imshow(image, cmap=cmap_type)
  plt.title(title)
  plt.axis('on')

def get_edges(image, low_threshold=85, high_threshold=95, step=1, edge_presence = 0.5, edge_detection='sobel'):
  image_grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  edges = []
  
  # Get the result through low and high thresholding to get real edges following the step
  for i in range(low_threshold, high_threshold, step):

    # Put the grayscale black and white only
    image_threshold = cv2.threshold(image_grayscale, i, 255, cv2.THRESH_BINARY)[1]

    if edge_detection == 'sobel':
      # Apply Sobel operator for edge detection
      x_res = cv2.Sobel(image_threshold, cv2.CV_64F, 1, 0, ksize=1)
      y_res = cv2.Sobel(image_threshold, cv2.CV_64F, 0, 1, ksize=1)
    else:
      # Apply Scharr operator for edge detection
      x_res = cv2.Scharr(image_threshold, cv2.CV_64F, 1, 0)
      y_res = cv2.Scharr(image_threshold, cv2.CV_64F, 0, 1)

    # Combine the x and y results to get the edges
    edges.append(cv2.magnitude(x_res, y_res))
  
  # For each pixel of the image list count the number of edges detected
  edges = np.sum(edges, axis=0)
    
  # get the edges presence
  edges_min_presence = (high_threshold - low_threshold) * (edge_presence)
  
  # Convert the data to grayscale between 0 and 255
  edges = cv2.normalize(edges, None, 0, (high_threshold - low_threshold), cv2.NORM_MINMAX, cv2.CV_8U)
  # save in txt file the edges
  np.savetxt('edges.txt', edges, fmt='%d')
  # if the pixel has more than edges_min_presence value, it is an edge
  edges[edges < edges_min_presence] = 0
  edges[edges >= edges_min_presence] = 255
  
  # Convert the data to grayscale between 0 and 255
  edges_normalized = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

  # negate the whole image
  edges_normalized = cv2.bitwise_not(edges_normalized)
  return edges_normalized

# Load your image in RGB colors
# image = cv2.imread('image2022.jpeg')
image = cv2.imread('image_zones_claires_remplacees.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# show_image(image, 'Original Image')

# Define contrast and brightness adjustment parameters
# alpha = 1.5  # Contrast factor
alpha = 1.5  # Contrast factor
# beta = 30    # Brightness factor
beta = 10    # Brightness factor

# Adjust the contrast and brightness
image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# show_two_images(image, 'Original Image', image_c, 'Contrast and Brightness Adjusted Image')

# Remove the noise
image = cv2.medianBlur(image, 3)

# show_two_images(image, 'Original Image', image_denoize, 'Denoized Image')

# Get the edges using edge detection
edges_normalized = get_edges(image)

# show_image(edges_normalized, 'Edges normalized')

# Replace with the threshold value you want
boat_threshold = 245
# sand_threshold = 10

# Apply the threshold
edges_normalized[edges_normalized < boat_threshold] = 0
edges_normalized[edges_normalized >= boat_threshold] = 255
# edges_normalized[(edges_normalized <= boat_threshold) & (edges_normalized > sand_threshold)] = 128

# Remove the noise
# edges_normalized = cv2.medianBlur(edges_normalized, 1)

# fuze the original image with the edges
superposed_image = image.copy()
# Remove the noise
# superposed_image = cv2.medianBlur(superposed_image, 5)

# If the pixel is not an edge, the pixel value is the mean of the original and the edge
superposed_image[edges_normalized == 255] = (superposed_image[edges_normalized == 255] + [0, 0, 0]) / 2
# If the pixel is an edge, the pixel value is black
superposed_image[edges_normalized == 0] = [0, 0, 0]
# superposed_image[edges_normalized == 128] = [255, 0, 0]

# Remove the noise
# superposed_image = cv2.medianBlur(superposed_image, 3)
# Show original image in color
# show_image(image, 'Original Image')

# show_image(edges, 'Sobel Edges')
# show_two_images(edges, 'Sobel Edges', edges_scharr, 'Scharr Edges')

# Show the original image and the edges
# show_two_images(edges, 'Edges', edges_normalized, 'Edges normalized')

# Save the image
cv2.imwrite('edges_normalized.jpg', edges_normalized)
cv2.imwrite('superposed_image.jpg', cv2.cvtColor(superposed_image, cv2.COLOR_RGB2BGR))