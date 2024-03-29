import cv2
import numpy as np

# def remove_shadows(image):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Apply adaptive thresholding to create a binary mask of shadows
#     mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 2)
    
#     # Invert the mask to highlight shadows
#     mask = cv2.bitwise_not(mask)
    
#     # Use morphology operations to fill gaps and remove small noise
#     kernel = np.ones((3, 3), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
#     # Apply the mask to the original image to remove shadows
#     result = cv2.bitwise_and(image, image, mask=mask)
    
#     return result

def remove_shadows(image):
  # Convert to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Thresholding to segment shadows
  _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

  # Apply median filtering to remove noise
  binary = cv2.medianBlur(binary, 5)
  
  cv2.imshow('Binary', binary)

  # Inpainting to fill shadow regions
  result = cv2.inpaint(image, binary, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

  return result

# Load your UAV image
image = cv2.imread('image2022.jpeg')

# Remove shadows
image_without_shadows = remove_shadows(image)

# Display the original and shadowless images
cv2.imshow('Original Image', image)
cv2.imshow('Shadowless Image', image_without_shadows)
cv2.waitKey(0)
cv2.destroyAllWindows()
