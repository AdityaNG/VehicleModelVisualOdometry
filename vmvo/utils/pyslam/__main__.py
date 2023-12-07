import math
import sys
import time

import cv2
import numpy as np
from pyslam.camera import PinholeCamera
from pyslam.config import Config
from pyslam.dataset import dataset_factory
from pyslam.feature_manager import feature_manager_factory
from pyslam.feature_matcher import FeatureMatcherTypes, feature_matcher_factory
from pyslam.feature_tracker import FeatureTrackerTypes, feature_tracker_factory
from pyslam.feature_tracker_configs import FeatureTrackerConfigs
from pyslam.feature_types import (
    FeatureDescriptorTypes,
    FeatureDetectorTypes,
    FeatureInfo,
)
from pyslam.ground_truth import groundtruth_factory

# from mplot3d import Mplot3d
# from mplot2d import Mplot2d
from pyslam.mplot_thread import Mplot2d, Mplot3d
from pyslam.visual_odometry import VisualOdometry

# from pyslam.visual_odometry import Visual_IMU_GPS_Odometry


def main(
    video_path,
    scale_factor=0.5,
    plot_3D_x=250,
    plot_3D_y=500,
    enable_plot=True,
):
    trajectory = {"x": [], "y": [], "z": [], "rot": []}

    cap = cv2.VideoCapture(video_path)
    img_id = 0
    ret, frame = cap.read()
    if ret == False:
        return

    cam_settings = {
        "Camera.width": frame.shape[1],
        "Camera.height": frame.shape[0],
        "Camera.fx": 1394.6027293299926,
        "Camera.fy": 1394.6027293299926,
        "Camera.cx": 995.588675691456,
        "Camera.cy": 599.3212928484164,
        "Camera.fps": 30,
    }
    # DistCoef = np.array([k1, k2, p1, p2, k3])
    DistCoef = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    cam = PinholeCamera(
        cam_settings["Camera.width"] * scale_factor,
        cam_settings["Camera.height"] * scale_factor,
        cam_settings["Camera.fx"] * scale_factor,
        cam_settings["Camera.fy"] * scale_factor,
        cam_settings["Camera.cx"] * scale_factor,
        cam_settings["Camera.cy"] * scale_factor,
        DistCoef,
        cam_settings["Camera.fps"],
    )
    num_features = 2000  # how many features do you want to detect and track?

    # select your tracker configuration (see the file feature_tracker_configs.py)
    # LK_SHI_TOMASI, LK_FAST
    # SHI_TOMASI_ORB, FAST_ORB, ORB, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, FAST_TFEAT
    tracker_config = FeatureTrackerConfigs.LK_SHI_TOMASI
    tracker_config["num_features"] = num_features

    feature_tracker = feature_tracker_factory(**tracker_config)
    print(feature_tracker)
    # create visual odometry object
    # vo = Visual_IMU_GPS_Odometry(cam, None, feature_tracker)
    vo = VisualOdometry(cam, None, feature_tracker)

    print("Computing Trajectory")
    plot_3D = np.zeros((plot_3D_x, plot_3D_y, 3))

    while True:
        ret, img_frame_rgb = cap.read()
        img_id += 1
        if ret == False:
            return

        img_frame = cv2.cvtColor(img_frame_rgb, cv2.COLOR_BGR2GRAY)
        img_frame_scaled = cv2.resize(
            img_frame, (0, 0), fx=scale_factor, fy=scale_factor
        )
        vo.track(img_frame_scaled, img_id)

        # vo.track(img_frame_scaled, img_id,
        #     accel_data=np.array([
        #         0.0, 0.0, 0.0
        #     ]).reshape((3,1)),
        #     gyro_data=np.array([
        #         0.0, 0.0, 0.0, # RotationV X, Y, Z
        #         0.0, 0.0, # RotationV W, Acc
        #     ]),
        #     gps_data=np.array([
        #         0.0, 0.0, 0.0, 0.0 # Lon, Lat, speed, heading
        #     ]),
        #     timestamp=time.time(),
        # )
        if img_id > 2:
            x, y, z = vo.traj3d_est[-1]
            rot = np.array(vo.cur_R, copy=True)
        else:
            # x, y, z = [0.0], [0.0], [0.0]
            x, y, z = 0.0, 0.0, 0.0
            rot = np.eye(3, 3)

        if type(x) != float:
            x = float(x[0])
        if type(y) != float:
            y = float(y[0])
        if type(z) != float:
            z = float(z[0])

        trajectory["x"] += [x]
        trajectory["y"] += [y]
        trajectory["z"] += [z]
        trajectory["rot"] += [rot]

        if enable_plot:
            p3x = int(x / 10 + plot_3D_x // 2)
            p3y = int(z / 10 + plot_3D_y // 2)
            if p3x in range(0, plot_3D_x) and p3y in range(0, plot_3D_y):
                plot_3D = cv2.circle(plot_3D, (p3y, p3x), 2, (0, 255, 0), 1)

        if enable_plot:
            cv2.imshow("plot_3D", plot_3D)
            cv2.imshow("Camera", vo.draw_img)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break


if __name__ == "__main__":
    main(sys.argv[1])
