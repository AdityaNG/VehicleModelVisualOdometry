import cv2
import numpy as np


class KalmanSLAM:
    def __init__(self, sensor_ids: list = [0, 1]):
        # frame_id is the current frame number
        self.frame_id = 0
        self.sensor_ids = sensor_ids
        self.num_sensors = len(sensor_ids)

        # transforms is a dictionary of lists of 4x4 matrices
        # Each key refers to a sensor
        self.transforms = {}
        for id in self.sensor_ids:
            self.transforms[id] = []

        self.kf_transforms = []

        # Initialize Kalman Filter parameters
        dynamParams = 16
        measureParams = 8
        self.kf = cv2.KalmanFilter(dynamParams, measureParams)
        self.kf.measurementMatrix = np.eye(measureParams, dynamParams).astype(
            np.float32
        )
        self.kf.processNoiseCov = 1e-3 * np.eye(
            dynamParams, dynamParams
        ).astype(np.float32)
        self.kf.measurementNoiseCov = 1e-1 * np.eye(
            measureParams, measureParams
        ).astype(np.float32)
        self.kf.errorCovPost = np.eye(dynamParams, dynamParams).astype(
            np.float32
        )
        self.kf.statePost = np.zeros((dynamParams, 1)).astype(np.float32)

    def track(self, transforms: dict):
        # transforms is a dictionary of 4x4 matrices
        # transforms[i] is the transformation matrix for sensor i

        for sensor_id, transform in transforms.items():
            assert sensor_id in self.sensor_ids
            assert isinstance(transform, np.ndarray)
            transforms[sensor_id] = transforms[sensor_id].astype(np.float32)

        # Kalman Filter
        final_transform = self.kf_func(transforms)
        self.kf_transforms.append(final_transform)
        self.frame_id += 1

    def kf_func(self, transforms: dict):
        # Stack the transformation matrices into a single array
        transform_array = []
        for sensor_id in self.sensor_ids:
            transform_array.append(transforms[sensor_id].reshape(-1, 1))
        transform_array = np.vstack(transform_array)

        # Predict step
        predicted_state = self.kf.predict()

        # Update step
        corrected_state = self.kf.correct(transform_array)

        # Extract the final transformation matrix from the corrected state
        final_transform = corrected_state[:4, :4].reshape(4, 4)

        return final_transform


# Demo code
if __name__ == "__main__":
    import math

    from .camera import PinholeCamera
    from .config import Config
    from .dataset import dataset_factory
    from .feature_tracker import feature_tracker_factory
    from .feature_tracker_configs import FeatureTrackerConfigs
    from .ground_truth import groundtruth_factory
    from .mplot_thread import Mplot2d, Mplot3d
    from .visual_odometry import VisualOdometry

    """
    use or not pangolin (if you want to use it then you need to install it by using the script install_thirdparty.sh)
    """
    kUsePangolin = False

    if kUsePangolin:
        from viewer3D import Viewer3D

    config = Config()

    dataset = dataset_factory(config.dataset_settings)

    groundtruth = groundtruth_factory(config.dataset_settings)

    cam = PinholeCamera(
        config.cam_settings["Camera.width"],
        config.cam_settings["Camera.height"],
        config.cam_settings["Camera.fx"],
        config.cam_settings["Camera.fy"],
        config.cam_settings["Camera.cx"],
        config.cam_settings["Camera.cy"],
        config.DistCoef,
        config.cam_settings["Camera.fps"],
    )

    num_features = 2000  # how many features do you want to detect and track?

    # select your tracker configuration (see the file feature_tracker_configs.py)
    # LK_SHI_TOMASI, LK_FAST
    # SHI_TOMASI_ORB, FAST_ORB, ORB, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, FAST_TFEAT
    tracker_config = FeatureTrackerConfigs.LK_SHI_TOMASI
    tracker_config["num_features"] = num_features

    feature_tracker = feature_tracker_factory(**tracker_config)

    # create visual odometry object
    # vo = VisualOdometry(cam, groundtruth, feature_tracker)
    vo = VisualOdometry(cam, None, feature_tracker)
    kf_slam = KalmanSLAM(sensor_ids=[0, 1])

    is_draw_traj_img = True
    traj_img_size = 800
    traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
    half_traj_img_size = int(0.5 * traj_img_size)
    draw_scale = 1

    is_draw_3d = True
    if kUsePangolin:
        viewer3D = Viewer3D()
    else:
        plt3d = Mplot3d(title="3D trajectory")

    is_draw_err = True
    err_plt = Mplot2d(xlabel="img id", ylabel="m", title="error")

    is_draw_matched_points = True
    matched_points_plt = Mplot2d(
        xlabel="img id", ylabel="# matches", title="# matches"
    )

    img_id = 0
    while dataset.isOk():
        img = dataset.getImage(img_id)

        if img is not None:
            vo.track(img, img_id)  # main VO function

            if (
                img_id > 2
            ):  # start drawing from the third image (when everything is initialized and flows in a normal way)
                x, y, z = vo.traj3d_est[-1]
                rot = np.array(vo.cur_R, copy=True)

                # convert (x, y, z) and rot to 4x4 transformation matrix
                T = np.eye(4)
                T[:3, :3] = rot
                T[:3, 3] = (x, y, z)
                kf_slam.track({0: T, 1: T})
                # kf_slam.track({0: T})
                T_KF = kf_slam.kf_transforms[-1]
                x, y, z = T_KF[:3, 3]
                rot = T_KF[:3, :3]

                x_true, y_true, z_true = vo.traj3d_gt[-1]

                if is_draw_traj_img:  # draw 2D trajectory (on the plane xz)
                    draw_x, draw_y = int(
                        draw_scale * x
                    ) + half_traj_img_size, half_traj_img_size - int(
                        draw_scale * z
                    )
                    true_x, true_y = int(
                        draw_scale * x_true
                    ) + half_traj_img_size, half_traj_img_size - int(
                        draw_scale * z_true
                    )
                    cv2.circle(
                        traj_img,
                        (draw_x, draw_y),
                        1,
                        (img_id * 255 / 4540, 255 - img_id * 255 / 4540, 0),
                        1,
                    )  # estimated from green to blue
                    cv2.circle(
                        traj_img, (true_x, true_y), 1, (0, 0, 255), 1
                    )  # groundtruth in red
                    # write text on traj_img
                    cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
                    text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
                    cv2.putText(
                        traj_img,
                        text,
                        (20, 40),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 255, 255),
                        1,
                        8,
                    )
                    # show
                    cv2.imshow("Trajectory", traj_img)

                if is_draw_3d:  # draw 3d trajectory
                    if kUsePangolin:
                        viewer3D.draw_vo(vo)
                    else:
                        plt3d.drawTraj(
                            vo.traj3d_gt, "ground truth", color="r", marker="."
                        )
                        plt3d.drawTraj(
                            vo.traj3d_est, "estimated", color="g", marker="."
                        )
                        plt3d.refresh()

                if is_draw_err:  # draw error signals
                    errx = [img_id, math.fabs(x_true - x)]
                    erry = [img_id, math.fabs(y_true - y)]
                    errz = [img_id, math.fabs(z_true - z)]
                    err_plt.draw(errx, "err_x", color="g")
                    err_plt.draw(erry, "err_y", color="b")
                    err_plt.draw(errz, "err_z", color="r")
                    err_plt.refresh()

                if is_draw_matched_points:
                    matched_kps_signal = [img_id, vo.num_matched_kps]
                    inliers_signal = [img_id, vo.num_inliers]
                    matched_points_plt.draw(
                        matched_kps_signal, "# matches", color="b"
                    )
                    matched_points_plt.draw(
                        inliers_signal, "# inliers", color="g"
                    )
                    matched_points_plt.refresh()

            # draw camera image
            cv2.imshow("Camera", vo.draw_img)

        # press 'q' to exit!
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        img_id += 1

    # print('press a key in order to exit...')
    # cv2.waitKey(0)

    if is_draw_3d:
        if not kUsePangolin:
            plt3d.quit()
        else:
            viewer3D.quit()
    if is_draw_err:
        err_plt.quit()
    if is_draw_matched_points is not None:
        matched_points_plt.quit()

    cv2.destroyAllWindows()
