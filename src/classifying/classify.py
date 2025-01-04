import cv2
from .utils import ImageClassifier, BoW


def classify_signal(image):

    bow = BoW()
    bow.load_vocabulary("classifying/vocabulary")
    classifier = ImageClassifier(bow)
    classifier.load("classifying/classifier")
    predicted_label_name = classifier.predict_image(image)
    return predicted_label_name


def draw_detection(img, rectangle, signal_type):
    x, y, xw, yh = rectangle
    cv2.rectangle(img, (x, y), (xw, yh), (0, 255, 0), 2)
    cv2.putText(
        img,
        signal_type,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )
