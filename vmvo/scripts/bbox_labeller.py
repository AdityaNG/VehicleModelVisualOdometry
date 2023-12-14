"""
Monocular 3D Object labelling tool

Allows the user to apply 3D bounding boxes to vehicles and pedestrians
"""
import os
from typing import Tuple

import cv2
import numpy as np
import torch
from pandas import Series

from vmvo.datasets.bdd.bdd_raw import AndroidDatasetIterator
from vmvo.datasets.bdd.helper import DATASET_DIR, DAYTIME_IDS
from vmvo.utils.bbox import plot_boxes_on_image_and_in_bev

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


def ask_save_before_exit(
    bbox_labels_path,
    bbox_labels,
    default,
):
    """
    OpenCV dialog
    "Are you sure you want to exit without saving?"
    y/n
    """
    loaded_bbox_labels = load_bbox_labels(bbox_labels_path, default)
    if np.array_equal(bbox_labels, loaded_bbox_labels):
        return True

    frame = np.zeros((100, 400, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        "Save and exit?",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "y: Yes",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "n: No",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    while True:
        cv2.imshow("Exit without saving?", frame)
        key = cv2.waitKey(0)
        if key == ord("y"):
            save_bbox_labels(bbox_labels_path, bbox_labels)
            break
        elif key == ord("n"):
            break


def process_frame(
    timestamp: int,
    csv_frame: Series,
    img_frame: np.ndarray,
    bbox_labels_path: str,
    previous_timestamp: int,
    target_det: TargetDetector,
):
    """Process a single frame"""
    targets_index = 0
    frame_scale = 0.25
    detector_threshold = 0.1
    P2 = np.array(
        [
            [1250.6, 0.000000e00, 978.4, 0.000000e00],
            [0.000000e00, 1254.8, 562.1, 0.000000e00],
            [0.000000e00, 0.000000e00, 1.000000e00, 0.000000e00],
        ]
    )
    K = P2[:3, :3]
    load_targets = False
    increment_ui = 0.3

    canvas = np.zeros_like(img_frame)
    plot_frame = img_frame.copy()
    frame = np.concatenate((plot_frame, canvas), axis=1)
    frame = cv2.resize(frame, (0, 0), fx=frame_scale, fy=frame_scale)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

    prev_bbox_labels_path = os.path.join(
        os.path.dirname(bbox_labels_path),
        f"{previous_timestamp}.npy",
    )
    if os.path.exists(bbox_labels_path):
        bbox_labels = load_bbox_labels(bbox_labels_path, None)
        default = bbox_labels
    elif previous_timestamp is not None and os.path.exists(
        prev_bbox_labels_path
    ):
        bbox_labels = load_bbox_labels(prev_bbox_labels_path, None)
        default = bbox_labels

    if bbox_labels is None:
        load_targets = True

    if load_targets:
        targets = target_det.get_2D_targets(
            img_frame,
            threshold=detector_threshold,
            valid_classes=valid_2d_classes,
            K=K,
        )
        if len(targets.index) == 0:
            print("No targets found")
            return "increment"
        default = np.array(
            targets["bbox_3d"].tolist(),
            dtype=np.float32,
        )
        bbox_labels = load_bbox_labels(bbox_labels_path, default)

    while True:
        if bbox_labels.shape[0] == 0:
            print("No targets found")
            print("Saving empty bbox labels")
            save_bbox_labels(bbox_labels_path, bbox_labels)
            return "increment"

        canvas = np.zeros_like(img_frame)
        plot_frame = img_frame.copy()

        selected_target = np.array(
            [
                bbox_labels[targets_index],
            ]
        )
        unselected_targets = np.delete(bbox_labels, targets_index, axis=0)

        plot_boxes_on_image_and_in_bev(
            selected_target,
            plot_frame,
            canvas,
            (0, 255, 0),
            P2,
            box_class_list=BOX_CLASSES,
        )

        plot_boxes_on_image_and_in_bev(
            unselected_targets,
            plot_frame,
            canvas,
            (153, 255, 51),
            P2,
            use_classwise_color=True,
            box_class_list=BOX_CLASSES,
        )

        frame = np.concatenate((plot_frame, canvas), axis=1)
        frame = cv2.resize(frame, (0, 0), fx=frame_scale, fy=frame_scale)

        # cv2.imshow(f"Frame {timestamp}", frame)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(0)
        if key == ord("]"):
            # Auto save
            save_bbox_labels(bbox_labels_path, bbox_labels)
            return "increment"
        elif key == ord("["):
            save_bbox_labels(bbox_labels_path, bbox_labels)
            return "decrement"
        elif key == ord("q"):
            ask_save_before_exit(bbox_labels_path, bbox_labels, default)
            cv2.destroyAllWindows()
            return "quit"
        elif key == ord("x"):
            save_bbox_labels(bbox_labels_path, bbox_labels)
        elif key == ord("n"):
            targets_index = (targets_index + 1) % len(bbox_labels)
        elif key == ord("p"):
            targets_index = (targets_index - 1) % len(bbox_labels)
        elif key == ord("w"):
            bbox_labels[targets_index][11] += increment_ui
        elif key == ord("s"):
            bbox_labels[targets_index][11] -= increment_ui
        elif key == ord("k"):
            bbox_labels[targets_index][12] += increment_ui
        elif key == ord("j"):
            bbox_labels[targets_index][12] -= increment_ui
        elif key == ord("u"):
            bbox_labels[targets_index][10] += increment_ui
        elif key == ord("i"):
            bbox_labels[targets_index][10] -= increment_ui
        elif key == ord("d"):
            bbox_labels[targets_index][9] += increment_ui
        elif key == ord("a"):
            bbox_labels[targets_index][9] -= increment_ui
        elif key == ord("r"):
            # Recompute target candidates
            targets = target_det.get_2D_targets(
                img_frame,
                threshold=detector_threshold,
                valid_classes=valid_2d_classes,
                K=K,
            )
            if len(targets.index) == 0:
                print("No targets found")
                return "increment"
            bbox_labels = np.array(
                targets["bbox_3d"].tolist(),
                dtype=np.float32,
            )
            default = bbox_labels
        elif key == ord("l"):
            # Load the previous frame's labels
            if os.path.exists(prev_bbox_labels_path):
                bbox_labels = load_bbox_labels(prev_bbox_labels_path, default)
        elif key == ord("b"):
            # Delete selected bbox
            bbox_labels = np.delete(bbox_labels, targets_index, axis=0)
            if len(bbox_labels) == 0:
                print("No targets found")
                return "increment"
            targets_index = targets_index % len(bbox_labels)
        elif key == ord("v"):
            # Create a new bbox
            bbox_labels = np.insert(
                # bbox_labels, targets_index, default[targets_index], axis=0
                bbox_labels,
                targets_index,
                bbox_labels[targets_index],
                axis=0,
            )
            if len(bbox_labels) == 0:
                print("No targets found")
                return "increment"
            targets_index = (targets_index + 1) % len(bbox_labels)


def main(dataset_id):
    """Main function"""
    folder_path = os.path.join(
        DATASET_DIR,
        dataset_id,
    )
    bbox_labels_folder = os.path.join(
        folder_path,
        "bbox_labels",
    )
    dataset = AndroidDatasetIterator(
        folder_path=folder_path,
        compute_trajectory=True,  # Load the VO trajectory
        invalidate_cache=False,  # Do not clear cache
    )
    target_det = TargetDetector()

    index = 0
    previous_timestamp = None

    while index in range(len(dataset)):
        csv_frame, img_frame = dataset[index]
        timestamp = int(csv_frame["Timestamp"])

        bbox_labels_path = os.path.join(
            bbox_labels_folder,
            f"{timestamp}.npy",
        )

        ret = process_frame(
            timestamp,
            csv_frame,
            img_frame,
            bbox_labels_path,
            previous_timestamp,
            target_det,
        )

        key = cv2.waitKey(1)
        if key == ord("0"):
            # Force Quit
            break

        if ret == "increment":
            index += 20
        elif ret == "decrement":
            index -= 20
        elif ret == "quit":
            break

        previous_timestamp = timestamp

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset ID",
        default=DAYTIME_IDS[2],
    )
    args = parser.parse_args()
    main(args.dataset)
