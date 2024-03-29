import cv2
import numpy as np

# Read the image
img = cv2.imread('image2022.jpeg')
# img = cv2.imread('image_zones_claires_remplacees.jpg')

def shadow_remove1(img):
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
    shadowremov = cv2.merge(result_norm_planes)
    return shadowremov#Shadow removal
# cv2.imwrite('after_shadow_remove1.jpg', shad)

def shadow_remove2(img):
    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to create a mask of shadows
    mask = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Invert the mask to get the shadows
    mask_inv = cv2.bitwise_not(mask)
    
    # Apply the mask to the original image to remove shadows
    result = cv2.bitwise_and(img, img, mask=mask_inv)
    
    return result
  
def shadow_remove3(img):
    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to create a mask of shadows
    _, mask = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)
    
    # Convert the mask to 3 channels to use it as a mask
    mask_3_channels = cv2.merge([mask, mask, mask])
    
    # Replace shadowed areas with a brighter tone
    result = cv2.add(img, mask_3_channels)
    
    return result

def shadow_remove4(img):
    # Convert image to LAB color space
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into channels
    l, a, b = cv2.split(lab_img)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel
    clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(3, 3))
    cl = clahe.apply(l)
    
    # Merge the CLAHE-enhanced L channel with the original A and B channels
    merged_lab = cv2.merge((cl, a, b))
    
    # Convert the merged LAB image back to BGR color space
    result = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    
    return result


def remove_high_a(img, threshold=150):
    # Convert image to LAB color space
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into channels
    l, a, b = cv2.split(lab_img)
    
    # Threshold the 'a' channel
    _, mask = cv2.threshold(a, threshold, 255, cv2.THRESH_BINARY)
    
    # Invert the mask
    mask_inv = cv2.bitwise_not(mask)
    
    # Set pixels with high 'a' component to black
    bgr_mask = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
    result = cv2.bitwise_and(img, bgr_mask)
    
    return result

# Remove pixels with high 'a' component
result = remove_high_a(img)
# shad = shadow_remove4(img)
cv2.imshow('Shadow', result)

cv2.waitKey(0)
cv2.destroyAllWindows()