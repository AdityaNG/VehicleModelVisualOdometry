"""
Monocular 3D Object labelling tool

Allows the user to apply 3D bounding boxes
"""
import base64
import os
from typing import Tuple

import cv2
import instructor
import numpy as np
from openai import OpenAI
from pandas import Series

from vmvo.datasets.bdd.bdd_raw import AndroidDatasetIterator
from vmvo.datasets.bdd.helper import DATASET_DIR, DAYTIME_IDS
from vmvo.schema import GPTLabel
from vmvo.utils.bbox import plot_boxes_on_image_and_in_bev
from vmvo.utils.bbox_labeller import (
    BOX_CLASSES,
    TargetDetector,
    load_bbox_labels,
    save_bbox_labels,
    valid_2d_classes,
)

GPT_PROMPT_1 = """
Following is a visual of a cuboids drawn around inanimate objects (no people \
or personally identifiable information) that have minor errors in them.
The selected cuboids is highlighted in green, the other cuboids are drawn on \
for reference.
A Birds-Eye-View is drawn on the right

cuboid:
(height={height},width={width},length={length},X={X},Y={Y},Z={Z},rot={rot})

Following are what the values signify:
 - rot: Orientation (radians) increasing counter-clockwise from BEV
 - X: Lateral Position (meters), increasing to the right
 - Y: Elevation (meters) increasing upwards
 - Z: Distance from camera (meters) increasing away from camera
 - (height, width, length): Dimensions (meters)

Describe the green bounding box:
 - Is it too big or too small or just perfect?
 - Is it too high or too low or just perfect?
 - Is it too far to the left or right or just perfect?
 - Is it too far or too close in or just perfect?
 - Is it rotated too much or too little or just perfect?
"""

GPT_PROMPT_2 = """Following is a description of cuboids that have minor \
errors in them.

cuboid:
(height={height},width={width},length={length},X={X},Y={Y},Z={Z},rot={rot})

Following are what the values signify:
 - rot: Orientation (radians) increasing counter-clockwise from BEV
 - X: Lateral Position (meters), increasing to the right
 - Y: Elevation (meters) increasing upwards
 - Z: Distance from camera (meters) increasing away from camera
 - (height, width, length): Dimensions (meters)

Following is a description of the boxes. Based on the following description, \
adjust the boxes to best fit the object.
In order to get significant adjustments in (X, Y, Z, rot), the minimum delta \
for these values is {min_increment}.
If the bounding box is good enough, you will raise the done flag in the output.
If there is no object and the target has been misidentified, you will raise \
the drop flag in the output in order to drop the label.

Description: {description}

You will produce your output in the following json format:

    "height":   h,
    "width":    w,
    "length":   l,
    "X":        x,
    "Y":        y,
    "Z":        z,
    "rot":      r,
    "done":     d,
    "drop":     dr,

"""


def encode_opencv_image(img):
    _, buffer = cv2.imencode(".jpg", img)
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")
    return jpg_as_text


class GPTVision:
    def __init__(self):
        # self.client = OpenAI()
        self.client = instructor.patch(OpenAI())

    def fine_tune(
        self,
        image: np.ndarray,
        bbox_3d: Tuple[float],
        num_iters: int = 5,
    ) -> GPTLabel:
        base64_image = encode_opencv_image(image)
        # bbox_3d
        #   0   1       2  3   4   5   6    7     8    9    10   11   12
        # (cls, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                    {
                        "type": "text",
                        "text": GPT_PROMPT_1.format(
                            height=bbox_3d[6],
                            width=bbox_3d[7],
                            length=bbox_3d[8],
                            X=bbox_3d[9],
                            Y=bbox_3d[10],
                            Z=bbox_3d[11],
                            rot=bbox_3d[12],
                        ),
                    },
                ],
            }
        ]
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=800,
        )

        desc = response.choices[0].message.content
        print(desc)

        gpt_label = self.client.chat.completions.create(
            # model="gpt-4-vision-preview",
            model="gpt-3.5-turbo",
            response_model=GPTLabel,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Generate JSON response",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": GPT_PROMPT_2.format(
                                height=bbox_3d[6],
                                width=bbox_3d[7],
                                length=bbox_3d[8],
                                X=bbox_3d[9],
                                Y=bbox_3d[10],
                                Z=bbox_3d[11],
                                rot=bbox_3d[12],
                                description=desc,
                                min_increment=0.3,
                            ),
                        },
                    ],
                },
            ],
            max_tokens=1000,
        )

        print(gpt_label)

        return gpt_label


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

    canvas = np.zeros_like(img_frame)
    plot_frame = img_frame.copy()
    frame = np.concatenate((plot_frame, canvas), axis=1)
    frame = cv2.resize(frame, (0, 0), fx=frame_scale, fy=frame_scale)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

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
    num_iters = 5
    iter_count = 0

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

        # cv2.imshow(f"Frame {timestamp}", frame)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.destroyAllWindows()
            return "quit"

        gpt_label = gpt.fine_tune(
            frame,
            selected_target[0],
            num_iters=(num_iters - iter_count),
        )

        if gpt_label.drop:
            # Delete selected bbox
            bbox_labels = np.delete(bbox_labels, targets_index, axis=0)
            if len(bbox_labels) == 0:
                print("No targets found")
                return "increment"
            targets_index = targets_index % len(bbox_labels)
            iter_count = 0
        elif gpt_label.done:
            # Save and increment
            #   0   1       2  3   4   5   6    7     8    9    10   11   12
            # (cls, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d)
            bbox_labels[targets_index][[6, 7, 8, 9, 10, 11, 12]] = [
                gpt_label.height,
                gpt_label.width,
                gpt_label.length,
                gpt_label.X,
                gpt_label.Y,
                gpt_label.Z,
                gpt_label.rot,
            ]
            save_bbox_labels(bbox_labels_path, bbox_labels)
            if len(bbox_labels) == 0:
                print("No targets found")
                return "increment"

            if targets_index + 1 >= len(bbox_labels):
                print("Last target reached")
                return "increment"
            targets_index = targets_index + 1
            iter_count = 0

        iter_count += 1
        if iter_count >= num_iters:
            print("Number of iterations exceeded")
            print("===" * 10)

            if len(bbox_labels) == 0:
                print("No targets found")
                return "increment"
            targets_index = targets_index % len(bbox_labels)


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
