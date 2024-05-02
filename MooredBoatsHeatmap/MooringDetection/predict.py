from ultralytics import YOLO
from PIL import Image
from src import utils
from metadata import extract_metadata
import time
import os
import pandas as pd

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

MODEL = os.path.join(SCRIPT_PATH, "model", "best.pt")

# IMAGES_DIR = os.path.join(SCRIPT_PATH, "data", "lerins", "2021 10 ZONE 2")
# IMAGES_DIR = os.path.join(SCRIPT_PATH, "data", "lerins", "2023 09 ZONE 2")
IMAGES_DIR = os.path.join(SCRIPT_PATH, "data", "lerins", "2021And2023_ZONE2")

# IMAGES_DIR = os.path.join(SCRIPT_PATH, "data", "images", "validationBackground")

# SOURCE = "data/lerins/2023 09 ZONE 2/Zone 2 2pm 1.JPG"
SOURCE = "data/lerins/2023 09 ZONE 2/Zone 2 2pm 4.JPG"
CSV_PREDICT_FILE_PATH = os.path.join(SCRIPT_PATH, "predictResults.csv")
CSV_NO_OVERLAPS_FILE_PATH = os.path.join(SCRIPT_PATH, "noOverlaps_predictResults.csv")


def predict(model: YOLO, img_path: str) -> list:
    # Replace backslash by slash
    img_path = img_path.replace("\\", "/")

    # Open the image
    originImage = Image.open(img_path)

    # Extract metadata from the picture
    exif_data = extract_metadata(img_path)
    exif_data = utils.get_clean_exif(exif_data)

    # predict on an image
    results = model.predict(
        source=img_path,
        save=True,
        imgsz=originImage.width,
        conf=0.75,
        line_width=5,
        # show_conf=True,
        # show_labels=False,
    )

    # Create a list to store the results
    results_list = []

    # Process results list
    for result in results:
        # Boxes object for box outputs
        boxes = result.boxes

        # Get the classes of the boxes
        classes = [int(cls.item()) for cls in boxes.cls]

        img = utils.get_image(result=result)

        for i in range(len(classes)):
            if classes[i] == 0:
                boxe_np = boxes[i].cpu().numpy()
                # Get boat coordinates
                coords = utils.extractLocalization(boxe_np, img, exif_data)

                # Store the result in the list
                result = [
                    classes[i],
                    exif_data["date"],
                    coords[0],
                    coords[1],
                    boxe_np.xyxy[0][0],
                    boxe_np.xyxy[0][1],
                    boxe_np.xyxy[0][2],
                    boxe_np.xyxy[0][3],
                    exif_data["orientation"],
                    img_path,
                ]
                results_list.append(result)

    return results_list


if __name__ == "__main__":
    # Keep start time
    start_time = time.time()

    # Get all images in the folder
    images_list = [path.path for path in os.scandir(IMAGES_DIR) if path.is_file()]

    nb_images = len(images_list)
    print(f"Found {nb_images} images.")

    # Load the model
    model = YOLO(MODEL)

    # Create a Pandas DataFrame to store the results
    predict_columns = [
        "class",
        "date",
        "lat",
        "lng",
        "boxe_top_left_x",
        "boxe_top_left_y",
        "boxe_bottom_right_x",
        "boxe_bottom_right_y",
        "orientation",
        "image",
    ]
    predictions = []

    for index, image_path in enumerate(images_list):
        # Run the prediction
        results = predict(model=model, img_path=image_path)

        # Add the results to the DataFrame
        predictions.extend(results)

        print(f"Image {index+1} / {nb_images} processed.")

    # Create and save the predict DataFrame to a CSV file
    predicts_df = pd.DataFrame(data=predictions, columns=predict_columns)
    # Set the date column as Datetime
    predicts_df["date"] = pd.to_datetime(predicts_df["date"])
    # Sort the DataFrame by date
    predicts_df = predicts_df.sort_values(by=["date"]).reset_index(drop=True)
    
    predicts_df.to_csv(CSV_PREDICT_FILE_PATH, index=True)
    
    # Remove the overlaps between images
    no_overlaps_df = utils.remove_overlap_between_images(predicts_df)

    # Save the clean DataFrame to a CSV file
    no_overlaps_df.to_csv(CSV_NO_OVERLAPS_FILE_PATH, index=False)
    
    # Keep end time
    end_time = time.time()

    # Compute execution time
    execution_time = end_time - start_time
    print(f"Code executed in {execution_time:.4f} seconds.")
