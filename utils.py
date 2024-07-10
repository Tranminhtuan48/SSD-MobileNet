import cv2
import numpy as np

# 20 class labels
CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
np.random.seed(111)
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
CONFIDENCE = 0.5


def dnn_detection_to_points(detection, width, height):
    x1 = int(detection[3] * width)
    y1 = int(detection[4] * height)
    x2 = int(detection[5] * width)
    y2 = int(detection[6] * height)

    return x1, y1, x2, y2


def draw_bounding_box_with_label(image, x1, y1, x2, y2, label, color, thickness=5):
    # bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    label_size, _ = cv2.getTextSize(label, font, font_scale, thickness=thickness)
    cv2.rectangle(
        image,
        (x1 - int(thickness / 2), y1 - label_size[1]),
        (x1 + label_size[0], y1),
        color,
        cv2.FILLED,
    )

    cv2.putText(
        image, label, (x1, y1), font, font_scale, color=(0, 0, 0)
    )
