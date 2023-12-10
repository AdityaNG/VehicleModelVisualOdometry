"""
DatasetHelper.py
    AndroidDatasetIterator
    PandaDatasetRecorder
"""

import os
from datetime import datetime, timedelta

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from .helper import DATASET_LIST, SETTINGS_DOC


class AndroidDatasetIterator:

    """
    AndroidDatasetIterator
    Iterates through dataset, given the folder_path
    """

    def __init__(
        self,
        folder_path=DATASET_LIST[-1],
        scale_factor=1.0,
        settings_doc=SETTINGS_DOC,
        invalidate_cache=False,
        compute_trajectory=False,
    ) -> None:
        print("Init path:", folder_path)
        assert os.path.exists(folder_path), (
            "Folder path does not exist: " + folder_path
        )
        self.folder_path = folder_path
        self.scale_factor = scale_factor
        self.old_frame_number = 0
        self.line_no = 0
        self.compute_trajectory = compute_trajectory

        self.id = folder_path.split("/")[-1]
        # self.start_time = int(self.id)
        self.csv_path = os.path.join(folder_path, self.id + ".csv")
        self.mp4_path = os.path.join(folder_path, self.id + ".mp4")

        assert os.path.exists(self.mp4_path), (
            "mp4_path file does not exist: " + self.mp4_path
        )

        # CSV stores time in ms
        self.csv_dat = pd.read_csv(self.csv_path)
        self.start_time = self.csv_dat["Timestamp"].iloc[0]
        self.csv_dat = self.csv_dat.sort_values("Timestamp")

        self.cap = cv2.VideoCapture(self.mp4_path)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        assert self.fps > 0, "FPS is not greater than 0"

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Computed video duration from FPS and number of video frames
        self.duration = self.frame_count / self.fps

        self.start_time_csv = min(self.csv_dat["Timestamp"])
        self.end_time_csv = max(self.csv_dat["Timestamp"])
        # Computed Duration the CSV file runs for
        self.expected_duration = (
            self.end_time_csv - self.start_time_csv
        ) / 1000.0

        self.csv_fps = len(self.csv_dat) / self.expected_duration

        # Expected FPS from CSV duration and number of frames
        self.expected_fps = self.frame_count / self.expected_duration
        # TODO: Perform Plausibility check on self.expected_fps and self.fps

        csv_final_frame_number = round(
            self.expected_duration * self.fps / 1000.0
        )
        # video_final_frame_number = round(self.duration * self.fps / 1000.0)

        assert csv_final_frame_number <= self.frame_count, (
            "csv_final_frame_number > self.frame_count"
            + str(csv_final_frame_number)
            + " "
            + str(self.frame_count)
        )

        self.length = 0
        for key in range(len(self.csv_dat)):
            timestamp = self.csv_dat.loc[key][0]
            time_from_start = timestamp - self.start_time_csv
            frame_number = round(time_from_start * self.fps / 1000.0)
            if frame_number < self.frame_count:
                self.length += 1
            else:
                break
        self.length = self.length - 1

        self.settings_doc = os.path.expanduser(settings_doc)
        with open(self.settings_doc, "r") as stream:
            try:
                self.cam_settings = yaml.load(stream, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                print(exc)
        k1 = self.cam_settings["Camera.k1"]
        k2 = self.cam_settings["Camera.k2"]
        p1 = self.cam_settings["Camera.p1"]
        p2 = self.cam_settings["Camera.p2"]
        k3 = 0
        if "Camera.k3" in self.cam_settings:
            k3 = self.cam_settings["Camera.k3"]
        self.DistCoef = np.array([k1, k2, p1, p2, k3])
        self.camera_matrix = np.array(
            [
                [
                    self.cam_settings["Camera.fx"],
                    0.0,
                    self.cam_settings["Camera.cx"],
                ],
                [
                    0.0,
                    self.cam_settings["Camera.fy"],
                    self.cam_settings["Camera.cy"],
                ],
                [0.0, 0.0, 1.0],
            ]
        )

        self.folder_path = os.path.dirname(self.csv_path)
        cached_trajectory_folder = os.path.join(
            self.folder_path,
        )
        os.makedirs(cached_trajectory_folder, exist_ok=True)
        self.cached_trajectory_path = os.path.join(
            cached_trajectory_folder,
            os.path.basename(self.csv_path).replace(".csv", "_traj.csv"),
        )
        if self.compute_trajectory:
            if (
                not os.path.exists(self.cached_trajectory_path)
                or invalidate_cache
            ):
                self.compute_slam()
            else:
                print(
                    "Loading trajectory from cache: ",
                    self.cached_trajectory_path,
                )
                # Load csv using pandas
                self.trajectory = pd.read_csv(self.cached_trajectory_path)

            def parse_rot(rot):
                # check if rot is a string
                if type(rot) == str:
                    rot = (
                        rot.replace("[", "").replace("]", "").replace("\n", "")
                    )
                    rot = rot.split()
                    rot = np.array(rot).astype(np.float32).reshape(3, 3)
                return rot

            self.trajectory["rot"] = self.trajectory["rot"].apply(parse_rot)
            self.trajectory["Timestamp"] = self.csv_dat["Timestamp"]
        else:
            self.trajectory = pd.DataFrame(
                {"x": [], "y": [], "z": [], "rot": []}
            )

    def __len__(self):
        # return len(self.csv_dat)
        return self.length - 30 * 5  # Buffer of 5 seconds to the end

    def __getitem__(self, key):
        if key > self.__len__():
            raise IndexError("Out of bounds; key=", key)
        timestamp = self.csv_dat.loc[key][0]
        time_from_start = timestamp - self.start_time_csv
        frame_number = round(time_from_start * self.fps / 1000.0)

        assert frame_number < self.frame_count, (
            "Frame number out of bounds: "
            + str(frame_number)
            + " >= "
            + str(self.frame_count)
        )

        delta = abs(frame_number - self.old_frame_number)
        if frame_number >= self.old_frame_number and delta < 5:
            for _ in range(delta - 1):
                ret, frame = self.cap.read()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            # print("cap.set: ", delta)

        self.old_frame_number = frame_number

        ret, frame = self.cap.read()
        if ret:
            w = int(frame.shape[1] * self.scale_factor)
            h = int(frame.shape[0] * self.scale_factor)
            final_frame = cv2.resize(frame, (w, h))
            return self.csv_dat.loc[key], final_frame

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            w = int(frame.shape[1] * self.scale_factor)
            h = int(frame.shape[0] * self.scale_factor)
            final_frame = cv2.resize(frame, (w, h))
            return self.csv_dat.loc[key], final_frame

        raise IndexError(
            "Frame number not catured: ",
            frame_number,
            ", key=",
            key,
            # File name
            self.mp4_path,
        )

    def compute_slam(
        self,
        scale_factor=0.25,
        enable_plot=False,
        plot_3D_x=250,
        plot_3D_y=500,
    ):
        # from pyslam.visual_imu_gps_odometry import Visual_IMU_GPS_Odometry
        from vmvo.utils.pyslam.camera import PinholeCamera
        from vmvo.utils.pyslam.feature_tracker import feature_tracker_factory
        from vmvo.utils.pyslam.feature_tracker_configs import (
            FeatureTrackerConfigs,
        )
        from vmvo.utils.pyslam.visual_odometry import VisualOdometry

        self.trajectory = {"x": [], "y": [], "z": [], "rot": []}

        cam = PinholeCamera(
            self.cam_settings["Camera.width"] * scale_factor,
            self.cam_settings["Camera.height"] * scale_factor,
            self.cam_settings["Camera.fx"] * scale_factor,
            self.cam_settings["Camera.fy"] * scale_factor,
            self.cam_settings["Camera.cx"] * scale_factor,
            self.cam_settings["Camera.cy"] * scale_factor,
            self.DistCoef,
            self.cam_settings["Camera.fps"],
        )
        num_features = (
            2000  # how many features do you want to detect and track?
        )

        # select your tracker configuration
        # (see the file feature_tracker_configs.py)
        # LK_SHI_TOMASI, LK_FAST
        # SHI_TOMASI_ORB, FAST_ORB, ORB, BRISK, AKAZE, FAST_FREAK, SIFT,
        # ROOT_SIFT, SURF, SUPERPOINT, FAST_TFEAT
        tracker_config = FeatureTrackerConfigs.LK_SHI_TOMASI
        tracker_config["num_features"] = num_features

        feature_tracker = feature_tracker_factory(**tracker_config)
        print(feature_tracker)
        # create visual odometry object
        self.vo = VisualOdometry(cam, None, feature_tracker)
        print("Computing Trajectory")
        plot_3D = np.zeros((plot_3D_x, plot_3D_y, 3))
        # start_id = self.__len__()-3
        start_id = 0
        end_offset = 50
        # for img_id in tqdm(range(0, self.__len__(), 1)):
        for img_id in tqdm(range(start_id, self.__len__() - end_offset, 1)):
            phone_frame = self.__getitem__(img_id)

            img_id = img_id - start_id

            # phone_data_frame, phone_img_frame = phone_frame
            _, phone_img_frame = phone_frame

            phone_img_frame_scaled = cv2.resize(
                phone_img_frame, (0, 0), fx=scale_factor, fy=scale_factor
            )

            self.vo.track(
                phone_img_frame_scaled,
                img_id,
            )
            if img_id > 2:
                x, y, z = self.vo.traj3d_est[-1]
                rot = np.array(self.vo.cur_R, copy=True)
            else:
                x, y, z = 0.0, 0.0, 0.0
                rot = np.eye(3, 3)

            if type(x) != float:
                x = float(x[0])
            if type(y) != float:
                y = float(y[0])
            if type(z) != float:
                z = float(z[0])

            self.trajectory["x"] += [x]
            self.trajectory["y"] += [y]
            self.trajectory["z"] += [z]
            self.trajectory["rot"] += [rot]

            if enable_plot:
                p3x = int(x / 10 + plot_3D_x // 2)
                p3y = int(z / 10 + plot_3D_y // 2)
                if p3x in range(0, plot_3D_x) and p3y in range(0, plot_3D_y):
                    plot_3D = cv2.circle(
                        plot_3D, (p3y, p3x), 2, (0, 255, 0), 1
                    )

            if enable_plot:
                cv2.imshow("plot_3D", plot_3D)
                cv2.imshow("Camera", self.vo.draw_img)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break

        for _ in range(end_offset):
            self.trajectory["x"] += [x]
            self.trajectory["y"] += [y]
            self.trajectory["z"] += [z]
            self.trajectory["rot"] += [rot]

        self.trajectory = pd.DataFrame(self.trajectory)
        self.trajectory.to_csv(self.cached_trajectory_path, index=False)

    def get_item_by_timestamp(self, timestamp, fault_delay=1000):
        """
        Return frame closest to given timestamp
        Raise exception if delta between timestamp and frame is
        greater than fault_delay
        """
        closest_frames = self.get_item_between_timestamp(
            timestamp - fault_delay,
            timestamp + fault_delay,
            fault_delay=float("inf"),
        )
        closest_frames = closest_frames.reset_index(drop=True)
        closest_frame = closest_frames.iloc[
            (closest_frames["Timestamp"] - timestamp).abs().argsort()[0]
        ]
        closest_ts = closest_frame["Timestamp"]
        if abs(timestamp - closest_ts) > fault_delay:
            raise Exception(
                "No such timestamp, fault delay exceeded:"
                + str(abs(timestamp - closest_ts))
            )

        closest_ts_index = self.csv_dat.index[
            self.csv_dat["Timestamp"] == closest_ts
        ].tolist()[0]
        return self.__getitem__(closest_ts_index)

    def get_item_between_timestamp(self, start_ts, end_ts, fault_delay=500):
        """
        Return frame between two given timestamps
        """
        ts_dat = self.csv_dat[
            self.csv_dat["Timestamp"].between(start_ts, end_ts)
        ]
        if len(ts_dat) == 0:
            raise Exception("No such timestamp")
        minimum_ts = min(ts_dat["Timestamp"])  # / 1000.0
        if abs(minimum_ts - start_ts) > fault_delay:
            raise Exception(
                "start_ts is out of bounds: abs(minimum_ts - start_ts)="
                + str(abs(minimum_ts - start_ts))
            )
        maximum_ts = max(ts_dat["Timestamp"])  # / 1000.0
        if abs(maximum_ts - end_ts) > fault_delay:
            raise Exception(
                "end_ts is out of bounds: abs(minimum_ts - start_ts)="
                + str(abs(maximum_ts - end_ts))
            )
        return ts_dat

    def __iter__(self):
        self.line_no = 0
        return self

    def __next__(self):
        data = self.__getitem__(self.line_no)
        self.line_no += 1
        return data

    def __str__(self) -> str:
        res = "----------------------------------------------------" + "\n"
        res += "AndroidDatasetIterator('" + self.folder_path + "')" + "\n"
        res += "----------------------------------------------------" + "\n"
        res += "self.fps:        \t" + str(self.fps) + "\n"
        res += "self.frame_count:\t" + str(self.frame_count) + "\n"
        res += (
            "self.start_time_csv:\t"
            + str(datetime.fromtimestamp(self.start_time_csv / 1000))
            + "\n"
        )
        res += (
            "self.end_time_csv:\t"
            + str(datetime.fromtimestamp(self.end_time_csv / 1000))
            + "\n"
        )
        res += (
            "self.expected_duration:\t"
            + str(timedelta(seconds=self.expected_duration))
            + "\n"
        )
        res += "self.expected_fps:\t" + str(self.expected_fps) + "\n"
        res += "self.csv_fps:        \t" + str(self.csv_fps) + "\n"
        res += "----------------------------------------------------"
        return res

    def __repr__(self) -> str:
        return str(self)

    def __del__(self):
        pass
