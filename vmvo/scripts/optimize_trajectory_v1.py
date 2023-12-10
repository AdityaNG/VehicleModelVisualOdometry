"""
Given a dataset ID, visualize the VO trajectory and the GPS trajectory
"""
import os

import cv2
import numpy as np

from vmvo.bicycle_model import BicycleModel
from vmvo.datasets.bdd.bdd_raw import AndroidDatasetIterator
from vmvo.datasets.bdd.helper import DATASET_DIR, DAYTIME_IDS
from vmvo.schema import Trajectory
from vmvo.utils.trajectory import (
    plot_bev_trajectory,
    plot_steering_traj,
    plot_trajectory_list,
    process_gps_trajectory,
    process_vo_trajectory,
)


def optimize_trajectory(
    vo_trajectory: Trajectory, gps_trajectory: Trajectory, model: BicycleModel
):
    """Optimize the trajectory using the bicycle model
    Iterate over each frame of the trajectory
    Take the average of the GPS and the VO trajectory
    Use the average as the desired trajectory
    """
    N = min(len(vo_trajectory), len(gps_trajectory))
    vmvo_trajectory = Trajectory(**dict(vo_trajectory))

    for i in range(N):
        vmvo_trajectory.x[i] = (vo_trajectory.x[i] + gps_trajectory.x[i]) / 2
        vmvo_trajectory.y[i] = (vo_trajectory.y[i] + gps_trajectory.y[i]) / 2

        # Calculate the average of theta considering it as an angle
        theta_diff = (vo_trajectory.theta[i] - gps_trajectory.theta[i]) % (
            2 * np.pi
        )
        if theta_diff > np.pi:
            theta_diff -= 2 * np.pi
        vmvo_trajectory.theta[i] = (
            vo_trajectory.theta[i] - theta_diff / 2
        ) % (2 * np.pi)

        vmvo_trajectory.velocity[i] = (
            vo_trajectory.velocity[i] + gps_trajectory.velocity[i]
        ) / 2

        assert np.isclose(
            vmvo_trajectory.time[i], vo_trajectory.time[i], atol=0.1
        ), (
            "Time mismatch: "
            + f"{vmvo_trajectory.time[i]} != {vo_trajectory.time[i]}"
        )

    return vmvo_trajectory


def main(dataset_id):
    """Main function"""
    folder_path = os.path.join(
        DATASET_DIR,
        dataset_id,
    )
    dataset = AndroidDatasetIterator(
        folder_path=folder_path,
        compute_trajectory=True,  # Load the VO trajectory
        invalidate_cache=False,  # Do not clear cache
    )
    # print("VO trajectory:")
    # print(dataset.trajectory)
    # print("=" * 10)
    # print("GPS trajectory:")
    # print(dataset.csv_dat)

    vo_trajectory = process_vo_trajectory(dataset.trajectory)
    gps_trajectory = process_gps_trajectory(dataset.csv_dat)

    print("VO trajectory:")
    print(vo_trajectory)
    print("=" * 10)
    print("GPS trajectory:")
    print(gps_trajectory)

    model = BicycleModel()

    vmvo_trajectory = optimize_trajectory(
        vo_trajectory,
        gps_trajectory,
        model,
    )

    plot_trajectory_list(
        [vo_trajectory, gps_trajectory, vmvo_trajectory],
        ["red", "green", "blue"],
        ["VO", "GPS", "VMVO"],
    )

    lookahead = 10.0  # s lookahead

    for index in range(len(dataset)):
        csv_frame, img_frame = dataset[index]
        frame = img_frame.copy()
        timestamp = csv_frame["Timestamp"] / 1000.0
        timestamp_next = timestamp + lookahead

        vo_trajectory_sub = vo_trajectory.sub_trajectory_from_time(
            timestamp, timestamp_next
        )

        gps_trajectory_sub = gps_trajectory.sub_trajectory_from_time(
            timestamp, timestamp_next
        )

        vmvo_trajectory_sub = vmvo_trajectory.sub_trajectory_from_time(
            timestamp, timestamp_next
        )

        frame = plot_steering_traj(
            frame,
            vo_trajectory_sub,
            color=(1, 0, 255),
        )
        frame = plot_steering_traj(
            frame,
            gps_trajectory_sub,
            color=(0, 255, 1),
        )
        frame = plot_steering_traj(
            frame,
            vmvo_trajectory_sub,
            color=(255, 0, 0),
        )
        vo_bev_frame = plot_bev_trajectory(
            frame,
            vo_trajectory_sub,
            color=(0, 0, 255),
        )
        gps_bev_frame = plot_bev_trajectory(
            frame,
            gps_trajectory_sub,
            color=(0, 255, 1),
        )
        vmvo_bev_frame = plot_bev_trajectory(
            frame,
            vmvo_trajectory_sub,
            color=(255, 0, 0),
        )

        # Overlay
        bev_frame = (
            vo_bev_frame * 0.3 + gps_bev_frame * 0.3 + vmvo_bev_frame * 0.3
        ).astype(np.uint8)

        frame = np.concatenate((frame, bev_frame), axis=1)

        frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
        cv2.imshow("Trajectory", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

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
