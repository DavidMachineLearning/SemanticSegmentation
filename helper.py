from PrePostProcessing import convert_prediction
import numpy as np
import cv2


def show_segmentation(image_, model):
    """Return the image with colored segmentation for a better visualization."""
    image_ = image_.reshape(1, 320, 800, 3)
    prediction = convert_prediction(model.predict(image_)) * 255
    masks = np.asarray(np.dstack((prediction[1], prediction[0], np.zeros((320, 800)))), dtype=np.int16)
    masks = masks.reshape((320, 800, 3))
    image_ = image_.reshape((320, 800, 3))
    return cv2.addWeighted(masks, 0.4, image_, 0.6, 1)
