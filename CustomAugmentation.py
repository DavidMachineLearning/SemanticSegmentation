from scipy.ndimage import rotate
import numpy as np
import cv2


def contrast_brightness(image_, contrast=1.0, brightness=0):
    """Return a given image with different contrast and brightness."""
    image_ = image_.copy()
    image_ = image_.astype(np.uint16)
    image_ = image_ * contrast + brightness
    return image_


def shift_image(image_, x=0, y=0):
    """Return a given image with horizontal (x) or vertical (y) shift (in pixels)."""
    image_ = np.roll(image_, y, axis=0)
    image_ = np.roll(image_, x, axis=1)
    if y > 0:
        image_[:y, :] = 0
    elif y < 0:
        image_[y:, :] = 0
    if x > 0:
        image_[:, :x] = 0
    elif x < 0:
        image_[:, x:] = 0
    return image_


def custom_generator(images_, masks_, batch_size=1):
    """Custom image generator for data augmentation.
    It can be used during the training process exactly like the one provided by Keras."""
    while True:
        h, w = images_.shape[1:3]
        center = (w / 2, h / 2)
        batch_indices = np.random.randint(0, images_.shape[0], size=batch_size)
        batch_x = np.zeros((batch_size, h, w, 3), dtype=np.int16)
        batch_y = np.zeros((batch_size, h * w, 2), dtype=np.int8)

        # image augmentation
        for i, index in enumerate(batch_indices):

            img = images_[index].copy()
            mask = masks_[index].copy().reshape((2, h, w))

            # randomly shift images and masks
            if np.random.random() < 0.5:
                shift = np.random.randint(-50, 51)
                img = shift_image(img, x=shift)
                mask[0] = shift_image(mask[0], x=shift)
                mask[1] = shift_image(mask[1], x=shift)

            # randomly flip images and masks
            if np.random.random() < 0.5:
                img = cv2.flip(img, 1)
                mask[0] = cv2.flip(mask[0], 1)
                mask[1] = cv2.flip(mask[1], 1)

            # randomly rotate images and masks
            if np.random.random() < 0.5:
                angle = np.random.randint(-30, 31)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

                img = cv2.warpAffine(img, matrix, (w, h))
                mask[0] = rotate(mask[0], angle, reshape=False)
                mask[1] = rotate(mask[1], angle, reshape=False)

            # randomly change contrast and brightness
            if np.random.random() < 0.5:
                img = contrast_brightness(img, contrast=np.random.randint(3, 11) / 10,
                                          brightness=np.random.randint(1, 76))

            # ensure pixels in the correct range
            img[img > 255] = 255
            img[img < 0] = 0
            mask[mask > 1] = 1
            mask[mask < 0] = 0

            batch_x[i] = img
            batch_y[i] = mask.reshape(h * w, 2)

        # Return a tuple of (image, mask) to feed the network
        yield (batch_x, batch_y)
