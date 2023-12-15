"""
Monocular 3D Object labelling tool

Allows the user to apply 3D bounding boxes to vehicles and pedestrians
"""
import os

import cv2
import numpy as np
from pandas import Series

from vmvo.datasets.bdd.bdd_raw import AndroidDatasetIterator
from vmvo.datasets.bdd.helper import DATASET_DIR, DAYTIME_IDS
from vmvo.utils.bbox import plot_boxes_on_image_and_in_bev
from vmvo.utils.bbox_labeller import (
    BOX_CLASSES,
    TargetDetector,
    load_bbox_labels,
    save_bbox_labels,
    valid_2d_classes,
)
from vmvo.utils.gpt import GPTVision


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
    gpt: GPTVision,
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

    bbox_labels = None
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
            gpt=gpt,
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

        unselected_bb_3d = np.zeros_like(plot_frame)

        plot_boxes_on_image_and_in_bev(
            unselected_targets,
            unselected_bb_3d,
            canvas,
            (153, 255, 51),
            P2,
            use_classwise_color=True,
            box_class_list=BOX_CLASSES,
        )

        # Vertical flip canvas
        canvas = cv2.flip(canvas, 0)

        # Add unselected_bb_3d
        plot_frame = cv2.addWeighted(
            plot_frame, 0.85, unselected_bb_3d, 0.45, 0
        )

        frame = np.concatenate((plot_frame, canvas), axis=1)
        frame = cv2.resize(frame, (0, 0), fx=frame_scale, fy=frame_scale)

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
                gpt=gpt,
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
    gpt = GPTVision()

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
            gpt,
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
