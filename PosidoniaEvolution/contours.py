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

def get_edges(image, low_threshold=85, high_threshold=95, step=1, edge_presence = 0.5, edge_detection='sobel', is_grayscale=False):
  if is_grayscale:
    image_grayscale = image
  else:
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
  # np.savetxt('edges.txt', edges, fmt='%d')
  # if the pixel has more than edges_min_presence value, it is an edge
  edges[edges < edges_min_presence] = 0
  edges[edges >= edges_min_presence] = 255
  
  # Convert the data to grayscale between 0 and 255
  edges_normalized = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

  # negate the whole image
  edges_normalized = cv2.bitwise_not(edges_normalized)
  return edges_normalized

def mean_brightness_contrast(image):
  # Convert the image to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  # Calculate the mean brightness and contrast
  mean_brightness = np.mean(gray)
  contrast = np.std(gray)
  
  print("Mean Brightness:", mean_brightness)
  print("Mean Contrast:", contrast)
  
  return mean_brightness, contrast

def adjust_brightness_contrast(image, target_brightness, target_contrast):
  # Compute the current brightness and contrast of the image
  current_brightness = np.mean(image)
  current_contrast = np.std(image)
  
  # Compute the adjustment values to reach the target values by looking at only pixels with high blue values
  brightness_adjustment = target_brightness - current_brightness
  contrast_adjustment = target_contrast / current_contrast
  
  # Apply the brightness and contrast adjustments
  adjusted_image = image.astype(np.float32) + brightness_adjustment
  adjusted_image = adjusted_image * contrast_adjustment
  
  # Clip the values to ensure they are within the valid range [0, 255]
  adjusted_image = np.clip(adjusted_image, 0, 255)
  
  # Convert the image back to the uint8 data type
  adjusted_image = adjusted_image.astype(np.uint8)
  
  return adjusted_image

def apply_edges(image, boat_threshold: int=245, edges_low_threshold: int = 85, edges_high_threshold: int = 95, edges_presence: float= 0.5, is_grayscale=False):
  
  if is_grayscale:
    black = 0
  else:
    # Convert the image to RGB colors
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    black = [0, 0, 0]
  # image = adjust_brightness_contrast(image, 100, 31)
  
  # # Get the mean brightness and contrast of the image
  # mean_brightness_contrast(image)
  
  # Remove the noise
  image = cv2.medianBlur(image, 3)

  # Get the edges using edge detection
  edges_normalized = get_edges(image, low_threshold=edges_low_threshold, high_threshold=edges_high_threshold, edge_presence=edges_presence, is_grayscale=is_grayscale)

  # Apply the threshold
  edges_normalized[edges_normalized < boat_threshold] = 0
  edges_normalized[edges_normalized >= boat_threshold] = 255
  # edges_normalized[(edges_normalized <= boat_threshold) & (edges_normalized > sand_threshold)] = 128

  # Remove the noise
  # edges_normalized = cv2.medianBlur(edges_normalized, 1)
  
  # Save the image of the edges
  cv2.imwrite('edges_normalized.jpg', edges_normalized)
  
  # fuze the original image with the edges
  superposed_image = image.copy()
  
  if is_grayscale:
    superposed_image[edges_normalized == 255] = (superposed_image[edges_normalized == 255])
  else:
    # If the pixel is not an edge, the pixel value is the mean of the original and the edge
    superposed_image[edges_normalized == 255] = (superposed_image[edges_normalized == 255] + black) / 2
  
  # If the pixel is an edge, the pixel value is black
  superposed_image[edges_normalized == 0] = black
  # superposed_image[edges_normalized == 128] = [255, 0, 0]
  
  if not is_grayscale:
    # Convert the image to BGR colors
    superposed_image = cv2.cvtColor(superposed_image, cv2.COLOR_RGB2BGR)

  return superposed_image
  
if __name__ == '__main__':
  # Load your image in RGB colors
  # image = cv2.imread('image2022.jpeg')
  # image = cv2.imread('lerins2019.jpeg')
  image = cv2.imread('image_zones_claires_remplacees.jpg')

  apply_edges(image, boat_threshold=100, edges_low_threshold=55, edges_high_threshold=59, edges_presence=0.5, is_grayscale=True)