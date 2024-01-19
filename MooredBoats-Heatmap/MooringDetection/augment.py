from yolo_data_augmentations.utils import run_yolo_augmentor, get_image_data
import albumentations as A

def get_transforms_train(rotation: int, vertical_flip_proba: int, horizontal_flip_proba: int) -> list:
  # CenterCrop and Resize already done after don't need to implement it
	return [
		A.Rotate(limit=(rotation, rotation), p=1),
		A.VerticalFlip(p=vertical_flip_proba),
		A.HorizontalFlip(p=horizontal_flip_proba),
		# A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0),
		# A.CLAHE(clip_limit=(0, 1), tile_grid_size=(8, 8), always_apply=True),
		# A.Resize(image_width, image_height)
	]

def get_transforms_val(vertical_flip_proba: int, horizontal_flip_proba: int) -> list:
	return [
		A.VerticalFlip(p=vertical_flip_proba),
		A.HorizontalFlip(p=horizontal_flip_proba),
	]
 
def get_transforms_val_to_train(x_min: int, y_min: int, x_max: int, y_max: int, rotation: int, vertical_flip_proba: int, horizontal_flip_proba: int) -> list:
	return [
   	A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, p=1),
    A.Rotate(limit=(rotation, rotation), p=1),
		A.VerticalFlip(p=vertical_flip_proba),
		A.HorizontalFlip(p=horizontal_flip_proba),
	]

def get_image_suffix(rotation: int = None, part: str = None, vertical_flip_proba: int = None, horizontal_flip_proba: int = None):
	img_suffix = ""
 
	if rotation is not None:
		img_suffix += "_" + str(rotation)
  
	if part is not None:
		img_suffix += "_" + part

	if vertical_flip_proba == 1:
		img_suffix += "_" + "VF"
  
	if horizontal_flip_proba == 1:
		img_suffix += "_" + "HF"

	return img_suffix

def apply_augment_train(rotation: int, vertical_flip_proba: int, horizontal_flip_proba: int):
	transforms = get_transforms_train(rotation=rotation, vertical_flip_proba=vertical_flip_proba, horizontal_flip_proba=horizontal_flip_proba)
	
	img_suffix = get_image_suffix(rotation=rotation, vertical_flip_proba=vertical_flip_proba, horizontal_flip_proba=horizontal_flip_proba)

	run_yolo_augmentor(transforms=transforms, img_max_size=640, img_suffix=img_suffix, is_drawn=True)
 
def apply_augment_val(vertical_flip_proba: int, horizontal_flip_proba: int):
	transforms = get_transforms_val(vertical_flip_proba=vertical_flip_proba, horizontal_flip_proba=horizontal_flip_proba)

	img_suffix = get_image_suffix(vertical_flip_proba=vertical_flip_proba, horizontal_flip_proba=horizontal_flip_proba)

	run_yolo_augmentor(transforms=transforms, img_max_size=None, img_suffix=img_suffix, is_drawn=True, img_extension="jpg")


def apply_augment_val_to_train(image_data: list, part: str, x_min: int, y_min: int, x_max: int, y_max: int, rotation: int, vertical_flip_proba: int, horizontal_flip_proba: int):
	transforms = get_transforms_val_to_train(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, rotation=rotation, vertical_flip_proba=vertical_flip_proba, horizontal_flip_proba=horizontal_flip_proba)

	img_suffix = get_image_suffix(rotation=rotation, part=part, vertical_flip_proba=vertical_flip_proba, horizontal_flip_proba=horizontal_flip_proba)

	run_yolo_augmentor(image_data=image_data, transforms=transforms, img_max_size=None, img_suffix=img_suffix, save_background=False, is_drawn=True, img_extension="jpg")
 
def augment_train_dataset():
	# rotations = [0, 90, 180, 270]
	rotations = [0, 90]
 
 	# apply augments on training dataset
	for rotation in rotations:
		apply_augment_train(rotation=rotation, vertical_flip_proba=0, horizontal_flip_proba=0)
		apply_augment_train(rotation=rotation, vertical_flip_proba=0, horizontal_flip_proba=1)
		apply_augment_train(rotation=rotation, vertical_flip_proba=1, horizontal_flip_proba=0)
		apply_augment_train(rotation=rotation, vertical_flip_proba=1, horizontal_flip_proba=1)
	
def augment_val_dataset():
	# apply augments on validation dataset
	apply_augment_val(vertical_flip_proba=0, horizontal_flip_proba=0)
	apply_augment_val(vertical_flip_proba=0, horizontal_flip_proba=1)
	apply_augment_val(vertical_flip_proba=1, horizontal_flip_proba=0)
	apply_augment_val(vertical_flip_proba=1, horizontal_flip_proba=1)
 
def augment_val_to_train_dataset(output_size: int = 640, image_width: int = 5472, image_height: int = 3648):
	# Calculate the number of divisions along width and height
	num_divisions_width = image_width // output_size
	num_divisions_height = image_height // output_size

	rotations = [0, 90]
 
	# Get images
	images_data = get_image_data()

	for image_data in images_data:
		for rotation in rotations:
			for i in range(num_divisions_width):
				for j in range(num_divisions_height):
					x_min = i * output_size
					y_min = j * output_size
					x_max = x_min + output_size
					y_max = y_min + output_size
					part = f"sub_{i}_{j}"
					apply_augment_val_to_train(image_data=image_data, part=part, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, rotation=rotation, vertical_flip_proba=0, horizontal_flip_proba=0)
					apply_augment_val_to_train(image_data=image_data, part=part, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, rotation=rotation, vertical_flip_proba=0, horizontal_flip_proba=1)
					apply_augment_val_to_train(image_data=image_data, part=part, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, rotation=rotation, vertical_flip_proba=1, horizontal_flip_proba=0)
					apply_augment_val_to_train(image_data=image_data, part=part, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, rotation=rotation, vertical_flip_proba=1, horizontal_flip_proba=1)

if "__main__" == __name__:
    # augment_train_dataset()
    # augment_val_dataset()
    augment_val_to_train_dataset()