from PIL import Image
from datetime import datetime
import numpy as np
import cv2
import re
from numpy import cos, sin, tan, arctan, arctan2, sqrt, pi, degrees, radians
import pandas as pd

EARTH_RADIUS = 6378137.0  # Earth radius in meters
MAVIC_2_PRO_FOV = 76.65 # mm https://www.dji.com/fr/mavic-2/info
OVERLAP_DISTANCE_THRESHOLD_IN_METERS = 15 # meters
OVERLAP_TIME_THRESHOLD_IN_MINUTES = 30 # minutes

def rotate_image(input_path, output_path, degrees):

    # Open the image file
    img = Image.open(input_path)

    # Rotate the image by the specified degrees
    rotated_img = img.rotate(degrees, expand=True)

    # Save the rotated image
    rotated_img.save(output_path)


def get_image(result):
    # Plotting results
    res_plotted = result.plot()
    res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    return Image.fromarray(res_plotted)


def get_box_center(data):
    # calculate x,y coordinate of center
    centerX = int(
        (data["dimensions"]["pos1"]["x"] + data["dimensions"]["pos2"]["x"]) / 2
    )
    centerY = int(
        (data["dimensions"]["pos1"]["y"] + data["dimensions"]["pos2"]["y"]) / 2
    )
    # # Convert data["img"] to a NumPy array
    # img_np = np.array(data["img"])

    # # Draw a center point
    # cv2.circle(img_np, (centerX, centerY), 5, (255, 0, 0), -1)

    # data["img"] = Image.fromarray(img_np)

    # data["img"].show()
    return (centerX, centerY)

# Compute the latitude or the longitude in float from exif data
def get_clean_coordinates(coord_field: str) -> float:
    # Use a regular expression to extract the relevant parts of the string
    match = re.match(r"(\d+) deg (\d+)' ([\d.]+)\" ([NSWE])", coord_field)
    
    if match:
        degrees, minutes, seconds, direction = match.groups()
        
        # Convert the numeric parts to float
        degrees = float(degrees)
        minutes = float(minutes)
        seconds = float(seconds)
        
        # Convert the minutes and seconds to fractions of a degree
        minutes /= 60
        seconds /= 3600
        
        # Apply the direction (N, S, E, W)
        if direction in ['S', 'W']:
            degrees = -degrees
        
        # Return the value in decimal degrees
        return degrees + minutes + seconds
    else:
        raise ValueError("Invalid GPS coordinates format")

def get_clean_date_format(date_str: str, actual_date_format: str) -> str:
    date_obj = datetime.strptime(date_str, actual_date_format)
    return date_obj.strftime("%Y-%m-%d %H:%M:%S")

def get_field_of_view(focal_length_field: str) -> float:
    focal_length = float(focal_length_field.split(" ")[0])
    
    # Specific to the Mavic 2 Pro
    sensor_width = 13.2 # mm
    sensor_height = 8.8 # mm
    sensor_diagonal = sqrt(sensor_width ** 2 + sensor_height ** 2)
    
    horizontal_fov = degrees(2 * arctan(sensor_width / (2 * focal_length)))
    vertical_fov = degrees(2 * arctan(sensor_height / (2 * focal_length)))
    fov = degrees(2 * arctan(sensor_diagonal / (2 * focal_length)))
    return (horizontal_fov, vertical_fov, fov)

def get_distance_between_two_points(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    # Convert degrees to radians
    lat1_rad = lat1 * pi / 180
    lat2_rad = lat2 * pi / 180
    
    # Compute the degrees deltas in radians
    delta_lat_rad = (lat2 - lat1) * pi / 180
    delta_lng_rad = (lng2 - lng1) * pi / 180

    # Compute the sin of the deltas 
    sin_delta_lat_half = sin(delta_lat_rad / 2)
    sin_delta_lon_half = sin(delta_lng_rad / 2)  

    # 
    a = sin_delta_lat_half**2 + cos(lat1_rad) * cos(lat2_rad) * sin_delta_lon_half**2
    c = 2 * arctan2(sqrt(a), sqrt(1 - a))

    # Compute the distance
    distance = EARTH_RADIUS * c  # in metres
    
    return distance

def check_overlap_detection(lat1: float, lng1: float, lat2: float, lng2: float) -> bool:
    # Get the distance between the two points
    distance = get_distance_between_two_points(lat1, lng1, lat2, lng2)
    
    # If the distance is less than the threshold and time difference is within the limit, the two points are considered to be the same
    if distance < OVERLAP_DISTANCE_THRESHOLD_IN_METERS:
        # print("Distance between the two points:", distance, "meters")
        return True
    
    return False


def remove_overlap_between_images(df: pd.DataFrame) -> pd.DataFrame:
    print("Removing overlaps between images...")
    
    # Make the list of overlapping rows
    overlapping_rows = []
    
    # Iterate over each pair of rows in the DataFrame
    for i in range(len(df)):
        # Skip the row if it is already in the list of overlapping rows
        if i in overlapping_rows:
            continue
        
        for j in range(i + 1, len(df)):
            # Get the image of the two rows
            image1, image2 = df.iloc[i]["image"], df.iloc[j]["image"]
            
            # If the two rows are in different images and the time difference is within the limit
            if image1 != image2:
                # Get the time of the two rows
                time1, time2 = df.iloc[i]["date"], df.iloc[j]["date"]
                
                # Get the time difference in minutes
                time_diff_minutes = (time2 - time1).seconds / 60.0
                
                # If the time difference is within the limit
                if time_diff_minutes < OVERLAP_TIME_THRESHOLD_IN_MINUTES:
                    
                    lat1, lng1 = df.iloc[i]["lat"], df.iloc[i]["lng"]
                    lat2, lng2 = df.iloc[j]["lat"], df.iloc[j]["lng"]
                    
                    if check_overlap_detection(lat1, lng1, lat2, lng2):
                        # print(f"Overlap detected between indices {i} and {j} in different images within {OVERLAP_TIME_THRESHOLD_IN_MINUTES} minutes")
                        # Keep the row we need to remove
                        overlapping_rows.append(df.index[j])
    
    # Remove the overlapping rows
    print("Number of overlapping boats removed :", len(overlapping_rows))
    clean_df = df.drop(overlapping_rows).reset_index(drop=True)
    
    return clean_df


def get_lat_lng(data):
    # Get the x and y of the box center
    x, y = get_box_center(data=data)

    img = data["image"]
    exif_data = data["exif"]

    # Image latitude
    img_lat = exif_data["latitude"]
    # Image longitude
    img_lng = exif_data["longitude"]
    
    # Image size
    width, height = img.size
    
    # Compute of real coodinates of the point
    delta_X = (x - width / 2) * exif_data["pixelSize"]
    delta_Y = (height / 2 - y) * exif_data["pixelSize"]
    
    # Convert orientation angle to radians
    drone_orientation_deg = exif_data["orientation"]
    drone_orientation_rad = np.radians(drone_orientation_deg)
    
    # Rotate the coordinates based on drone orientation
    rotated_delta_X = delta_X * cos(drone_orientation_rad) - delta_Y * sin(drone_orientation_rad)
    rotated_delta_Y = delta_X * sin(drone_orientation_rad) + delta_Y * cos(drone_orientation_rad)

    # Compute the latitude and longitude of the point
    lat = img_lat + (rotated_delta_Y / EARTH_RADIUS * (180 / pi))
    lng = img_lng + (rotated_delta_X / (EARTH_RADIUS * cos(img_lat * pi / 180)) * (180 / pi))

    return (lat, lng)


def extractLocalization(boxe, img, exif_data: dict):
    dimensions = boxe.xyxy[0] # Get the boxes dimensions

    data = {
        "image": img,
        "exif": exif_data,
        "conf": boxe.conf[0],
        "dimensions": {
            "pos1": {
                "x": dimensions[0],
                "y": dimensions[1],
            },
            "pos2": {
                "x": dimensions[2],
                "y": dimensions[3],
            },
        },
    }
    # print("lat : ", y2lat(centerY))
    # print("lng : ", x2lng(centerX))
    return get_lat_lng(data=data)

def get_clean_exif(exif_data: dict) -> dict:
    # Get the useful exif data
    exif_clean_data = {
        "date" : get_clean_date_format(exif_data["CreateDate"], "%Y:%m:%d %H:%M:%S"),
        "latitude" : get_clean_coordinates(exif_data["GPSLatitude"]),
        "longitude" : get_clean_coordinates(exif_data["GPSLongitude"]),
        "altitude" : float(exif_data["AbsoluteAltitude"]),
        "fov" : get_field_of_view(exif_data["FocalLength"]),
        "orientation" : -float(exif_data["GimbalYawDegree"]),
    }
    # print("FOV: horizontal =", exif_clean_data["fov"][0], "vertical =", exif_clean_data["fov"][1], "diagonal =", exif_clean_data["fov"][2])
    # Image width in meters
    exif_clean_data["imageWidthInMeters"] = 2 * (tan(MAVIC_2_PRO_FOV / 2) * exif_clean_data["altitude"])
    # print("Image width in meters:", exif_clean_data["imageWidthInMeters"])
    
    # Real size of the pixel in the picture (in meter by pixel)
    exif_clean_data["pixelSize"] = exif_clean_data["imageWidthInMeters"] / exif_data["ImageWidth"]
    # print("Pixel size (m):", exif_clean_data["pixelSize"])
    
    return exif_clean_data