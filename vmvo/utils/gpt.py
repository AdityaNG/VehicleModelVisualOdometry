"""
GPT Vision to make fine tuning 3D bounding box labels easier
"""
import base64
from typing import Tuple

import cv2
import instructor
import numpy as np
from openai import OpenAI

from vmvo.schema import GPTLabel, GPTOrientation

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

GPT_ORIENTATION_PROMPT_1 = """Following is a snippet of an object in an image.\
The object is oriented in a certain way. Describe the orientation of the \
object in the image.

 - Is it facing right, left, forward, backward?
 - Note that backward means the object is facing the camera\
and forward means the object is facing away from the camera
 -Left Facing
 -Backward Left Facing
 -Backward Facing
 -Backward Right Facing
 -Right Facing
 -Forward Right Facing
 -Forward Facing
 -Forward Left Facing

Feel free to pick intermediate values
Orientation:
"""

GPT_ORIENTATION_PROMPT_2 = (
    f"""Following is a description an object in an image.
The object is oriented in a certain way. Provide the orientation theta of the \
object in the image when seen from a birds eye view in radians.

 - 0: ({0.0000}) Left Facing
 - pi/4:({round(np.pi/4, 4)})  Forward Left Facing
 - pi/2:({round(np.pi/2, 4)})  Forward Facing
 - 3pi/4:({round(3*np.pi/4, 4)})  Forward Right Facing
 - pi: ({round(np.pi, 4)}) Right Facing
 - 5pi/4:({round(5*np.pi/4, 4)})  Backward Right Facing
 - 3pi/2:({round(3*np.pi/2, 4)})  Backward Facing
 - 7pi/4:({round(7*np.pi/4, 4)})  Backward Left Facing

Be specific about the orientation, feel free to pick intermediate values
"""
    + """
Description:
{description}

Orientation:
"""
)


def encode_opencv_image(img):
    _, buffer = cv2.imencode(".jpg", img)
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")
    return jpg_as_text


class GPTVision:
    def __init__(self):
        self.client = instructor.patch(OpenAI())

    def guess_orientation(self, image: np.ndarray) -> float:
        base64_image = encode_opencv_image(image)
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
                        "text": GPT_ORIENTATION_PROMPT_1,
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
            response_model=GPTOrientation,
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
                            "text": GPT_ORIENTATION_PROMPT_2.format(
                                description=desc,
                            ),
                        },
                    ],
                },
            ],
            max_tokens=1000,
        )

        print(gpt_label)

        return gpt_label.theta

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
                            "text": GPT_PROMPT_1.format(
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
