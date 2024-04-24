import cv2
import os
import pandas as pd
import numpy as np
import time

EARTH_RADIUS = 6378137.0  # Radius of the Earth in meters

# Parameters for the google earth image of the Lerins islands
IMG_CENTER_LAT = 43.51251388888889
IMG_CENTER_LNG = 7.046827777777778
DRONE_ALTITUDE = 1600
DRONE_FOV = 58.5

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
# IMG_PATH = os.path.join(SCRIPT_PATH, "..", "PosidoniaEvolution", "superposed_image.jpg")
IMG_PATH = os.path.join(SCRIPT_PATH, "..", "PosidoniaEvolution", "images", "superposed_image_no_yellow.png")
BOATS_GPS_PATH = os.path.join(SCRIPT_PATH, "..", "MooredBoatsHeatmap", "MooringDetection", "noOverlaps_predictResults.csv")

# From the GPS coordinates, place pixels on the image
def gps_to_pixel(gps_coords, center_gps, altitude, fov, img_width, img_height):

    # Extract GPS coordinates
    lat1, lon1 = center_gps
    lat2, lon2 = gps_coords

    # Convert latitude and longitude from degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Calculate distance between the two GPS coordinates using Haversine formula
    d_lat = lat2_rad - lat1_rad
    d_lon = lon2_rad - lon1_rad
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = EARTH_RADIUS * c

    # Calculate the bearing angle from the center of the image to the GPS coordinates
    y = np.sin(lon2_rad - lon1_rad) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(lon2_rad - lon1_rad)
    bearing = np.arctan2(y, x)
    bearing = (bearing + 2 * np.pi) % (2 * np.pi)  # Normalize to [0, 2*pi]

    # Convert distance to pixels using altitude and field of view (FOV)
    image_width_in_meters = 2 * (np.tan(np.radians(fov / 2)) * altitude)
    pixel_size = image_width_in_meters / img_width
    distance_in_pixels = distance / pixel_size

    # Calculate pixel offsets
    x_offset = distance_in_pixels * np.sin(bearing)
    y_offset = distance_in_pixels * np.cos(bearing)

    # Calculate pixel coordinates relative to the center of the image
    center_x = img_width / 2
    center_y = img_height / 2
    pixel_x = center_x + x_offset
    pixel_y = center_y - y_offset  # Note: Y-axis is inverted in image coordinates

    return int(pixel_x), int(pixel_y)

if __name__ == '__main__':
    
    # Get the image with the posidonia evolution over time
    img = cv2.imread(IMG_PATH)
    img_width = img.shape[1]  # Width of the image
    img_height = img.shape[0]  # Height of the image
    print("Image width:", img_width, "| Image height:", img_height)

    # Example usage:
    center_gps = (IMG_CENTER_LAT, IMG_CENTER_LNG)  # GPS coordinates of the center of the image
    
    # read the csv file containing the GPS coordinates of the boats
    boat_gps_df = pd.read_csv(BOATS_GPS_PATH)
    
    # Draw the heatmap of pixel coordinates using sns.heatmap
    heatmap = np.zeros((img_height, img_width))
    radius = 4  # Radius of the circle to draw around each boat in pixels
    
    # for each boat, convert the GPS coordinates to pixel coordinates and draw a circle on the image
    for index, row in boat_gps_df.iterrows():
        gps_coords = (row["lat"], row["lng"])
        pixel_x, pixel_y = gps_to_pixel(gps_coords, center_gps, DRONE_ALTITUDE, DRONE_FOV, img_width, img_height)
        
        # Increase the count of boats in the pixel with a radius
        for x in range(pixel_x - radius, pixel_x + radius):
            for y in range(pixel_y - radius, pixel_y + radius):
                if 0 <= x < img_width and 0 <= y < img_height:
                    heatmap[y, x] += 1
    
    # Normalize the heatmap to be between 0 and 1
    heatmap = heatmap / np.max(heatmap)
    
    # Save the heatmap as an image with a colormap
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Save the heatmap image
    cv2.imwrite(os.path.join(SCRIPT_PATH, "heatmap.png"), heatmap)
    
    # Get the most common color in the heatmap
    unique_colors, counts = np.unique(heatmap.reshape(-1, 3), axis=0, return_counts=True)
    background_color = unique_colors[np.argmax(counts)]
    
    # Apply the heatmap to the original image without the background color of the heatmap with full opacity
    img_with_heatmap = img.copy()
    img_with_heatmap[heatmap != background_color] = heatmap[heatmap != background_color]
        
    # Save the image with the circle drawn
    cv2.imwrite(os.path.join(SCRIPT_PATH,"correlation_image.png"), img_with_heatmap)