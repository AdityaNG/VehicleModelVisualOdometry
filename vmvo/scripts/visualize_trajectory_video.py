"""
Given a dataset ID, visualize the VO trajectory and the GPS trajectory
"""
import os

import cv2
import numpy as np

from vmvo.datasets.bdd.bdd_raw import AndroidDatasetIterator
from vmvo.datasets.bdd.helper import DATASET_DIR, DAYTIME_IDS
from vmvo.utils.trajectory import (
    plot_bev_trajectory,
    plot_steering_traj,
    process_gps_trajectory,
    process_vo_trajectory,
)


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
    print("VO trajectory:")
    print(dataset.trajectory)
    print("=" * 10)
    print("GPS trajectory:")
    print(dataset.csv_dat)

    vo_trajectory = process_vo_trajectory(dataset.trajectory)
    gps_trajectory = process_gps_trajectory(dataset.csv_dat)

    print("VO trajectory:")
    print(vo_trajectory)
    print("=" * 10)
    print("GPS trajectory:")
    print(gps_trajectory)

    # plot_trajectory_list(
    #     [vo_trajectory, gps_trajectory],
    #     ["red", "green"],
    #     ["VO", "GPS"],
    # )
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

        # Overlay
        bev_frame = (vo_bev_frame * 0.5 + gps_bev_frame * 0.5).astype(np.uint8)

        frame = np.concatenate((frame, bev_frame), axis=1)

        print(vo_trajectory_sub[0])

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
