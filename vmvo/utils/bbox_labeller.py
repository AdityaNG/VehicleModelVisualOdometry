"""
Monocular 3D Object labelling tool

Allows the user to apply 3D bounding boxes to vehicles and pedestrians
"""
import os
from typing import Tuple

import numpy as np
import torch

valid_2d_classes = ("person", "bicycle", "car", "motorcycle", "bus", "truck")

bbox_2d_to_3d = {
    "person": "pedestrian",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
}


class_3d_dimensions = {
    "pedestrian": (0.5, 0.5, 1.5),
    "bicycle": (1.0, 1.0, 1.5),
    "car": (1.5, 2.0, 4.5),
    "motorcycle": (1.0, 1.0, 1.5),
    "cyclist": (1.0, 1.0, 1.5),
    "bus": (2.0, 2.0, 3.0),
    "truck": (2.0, 2.0, 3.0),
}

class_3d = {
    "pedestrian": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "cyclist": 4,
    "bus": 5,
    "truck": 6,
}

BOX_CLASSES = list(class_3d.keys())


def fit_3D_bbox(
    bbox_2d: Tuple[float, float, float, float],
    bbox_3d_dims: Tuple[float, float, float],
    cls: int,
    K: np.ndarray,
    elevation: float,
) -> Tuple[float]:
    """
    Given:
      position and dims of the 2D bounding box on the camera plane
      dims of the 3D bounding box
      camera intrinsics
      3D elevation of the camera from the ground plane
    Compute the 3D position of the 3D bounding box from the camera such that
      it fits the 2D bounding box on the camera plane'

    Assume that the object is sitting on the ground plane
    """
    xmin, ymin, xmax, ymax = bbox_2d
    width, height, length = bbox_3d_dims

    # Compute the center of the 2D bounding box
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2

    bbox_2d_width = xmax - xmin

    # Compute the focal length and optical center from the camera intrinsics
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Compute the distance of the 3D bounding box from the camera
    Z = width * fx / bbox_2d_width

    # Compute the 3D position of the 3D bounding box from the camera
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy - height / 2 - elevation

    # Compute the 3D position of the 3D bounding box from the camera
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy

    # Swap Axes
    # X, Y, Z = Y, Z, X
    # X, Y, Z = X, Z, Y

    alpha = 0
    ry3d = np.pi / 2

    # Compute the 3D bounding box as a tuple of 12 floats
    #   0   1       2  3   4   5   6    7     8    9    10   11   12
    # (cls, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d)
    bbox_3d = (
        cls,
        alpha,
        xmin,
        ymin,
        xmax,
        ymax,
        height,
        width,
        length,
        X,
        Y,
        Z,
        ry3d,
    )

    return bbox_3d


class TargetDetector:
    def __init__(
        self, model_name="ultralytics/yolov5", model_version="yolov5x6"
    ):
        """
        Initialize the TargetDetector with a YOLO model
        """
        self.model = torch.hub.load(model_name, model_version)

    @torch.no_grad()
    def get_2D_targets(
        self,
        img_frame: np.ndarray,
        threshold: float,
        valid_classes: Tuple[str],
        K: np.ndarray,
    ):
        """
        Use YOLO to select target candidates
        Look for pedestrians (people) and vehicles (car, truck, bus, etc)
        """
        img_rgb = img_frame[..., ::-1]  # OpenCV image (BGR to RGB)

        results = self.model(img_rgb, size=640)  # includes NMS
        results = results.pandas().xyxy[0]

        # Filter out targets with confidence less than threshold
        results = results[results["confidence"] > threshold]

        # Filter out targets that are not pedestrians or vehicles
        results = results[results["name"].isin(valid_classes)]

        results["bbox_3d_class"] = results["name"].apply(
            lambda name: bbox_2d_to_3d[name]
        )

        print("=" * 10)
        print(results)
        print("=" * 10)

        # Check if number of results>0
        if len(results.index) > 0:
            # Fit 3D bounding boxes
            results["bbox_3d"] = results.apply(
                lambda row: fit_3D_bbox(
                    (row["xmin"], row["ymin"], row["xmax"], row["ymax"]),
                    class_3d_dimensions[row["bbox_3d_class"]],
                    class_3d[row["bbox_3d_class"]],
                    K,
                    1.3,
                ),
                axis=1,
            )

        return results


def load_bbox_labels(bbox_labels_path, default):
    """
    Load the bounding box labels from the given path
    """
    if os.path.exists(bbox_labels_path):
        bbox_labels = np.load(bbox_labels_path)
        if bbox_labels.shape[0] == 0:
            bbox_labels = default
    else:
        bbox_labels = default
    return bbox_labels


def save_bbox_labels(bbox_labels_path, bbox_labels):
    """
    Save the bounding box labels to the given path
    """
    # Make parent directories if they don't exist
    os.makedirs(os.path.dirname(bbox_labels_path), exist_ok=True)
    np.save(bbox_labels_path, bbox_labels)
    print(f"Saved {bbox_labels_path}")
