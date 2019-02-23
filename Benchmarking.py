from PrePostProcessing import convert_prediction
from time import time


def accuracy_mean_iou(test_set, labels, model1, model2, model3):
    """Evaluate the accuracy of the 3 models."""
    model1_score = model1.evaluate(test_set, labels, batch_size=4, verbose=0)[1]
    model2_score = model2.evaluate(test_set, labels, batch_size=4, verbose=0)[1]
    model3_score = model3.evaluate(test_set, labels, batch_size=4, verbose=0)[1]
    return model1_score, model2_score, model3_score


def speed(test_set, model1, model2, model3):
    """Return the time needed by the 3 models to convert the provided images."""
    n_images = test_set.shape[0]
    start = time()
    convert_prediction(model1.predict(test_set, batch_size=4), output_shape=(n_images, 2, 320, 800))
    time1 = time() - start
    start = time()
    convert_prediction(model2.predict(test_set, batch_size=4), output_shape=(n_images, 2, 320, 800))
    time2 = time() - start
    start = time()
    convert_prediction(model3.predict(test_set, batch_size=4), output_shape=(n_images, 2, 320, 800))
    time3 = time() - start
    return time1, time2, time3
