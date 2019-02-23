import numpy as np


def convert_prediction(prediction_, threshold=0.5, output_shape=(2, 320, 800)):
    """Convert the output of the model in the actual prediction."""
    prediction_ = (prediction_ > threshold).astype(np.int8)
    return prediction_.reshape(output_shape)


def convert_segmentation(image_):
    """Used for a better visualization of the segmentation dataset."""
    height, width = image_.shape[:2]
    return np.vstack(image_).max(axis=1).reshape(height, width, 1)


def preprocess(image_, area=(160, 480, 0, 800)):
    """Crop the image to remove non relevant areas"""
    height = area[1] - area[0]
    width = area[3] - area[2]
    image_ = image_[area[0]:area[1], area[2]:area[3]]
    return image_.reshape(height, width, 3)


def create_masks(mask_, area=(160, 480, 0, 800)):
    """Pre-process step for masks, the mask get cropped like the image
    and only the masks related to road (pixel = 7 or 6) and cars (pixel = 10)
    are used."""
    height = area[1] - area[0]
    width = area[3] - area[2]
    mask_ = mask_[area[0]:area[1], area[2]:area[3]]
    output_masks = []

    mask_road = np.zeros((height, width), dtype=np.int8)
    mask_road[np.where(mask_ == 7)[0], np.where(mask_ == 7)[1]] = 1
    mask_road[np.where(mask_ == 6)[0], np.where(mask_ == 6)[1]] = 1
    output_masks.append(mask_road)

    mask_cars = np.zeros((height, width), dtype=np.int8)
    mask_cars[np.where(mask_ == 10)[0], np.where(mask_ == 10)[1]] = 1
    output_masks.append(mask_cars)

    return np.asarray(output_masks).reshape((height, width, 2))


def car_present(mask_, percentage=5):
    """Check if in the image there are 5% of pixels associated to cars.
    This is used to discard images where there are no cars."""
    mask_ = mask_[200:480, 0:600]
    mask_i = np.zeros((280, 600), dtype=np.int8)
    mask_i[np.where(mask_ == 10)[0], np.where(mask_ == 10)[1]] = 1
    unique, counts = np.unique(mask_i, return_counts=True)
    if 1 in unique:
        index = int(np.where(unique == 1)[0])
        if counts[index] * 100 / counts.sum() > percentage:
            return True
    return False
