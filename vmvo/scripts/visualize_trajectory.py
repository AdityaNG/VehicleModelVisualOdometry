"""
Given a dataset ID, visualize the VO trajectory and the GPS trajectory
"""
import os

from vmvo.datasets.bdd.bdd_raw import AndroidDatasetIterator
from vmvo.datasets.bdd.helper import DATASET_DIR, DAYTIME_IDS
from vmvo.utils.trajectory import (
    plot_trajectory_list,
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

    plot_trajectory_list(
        [vo_trajectory, gps_trajectory],
        ["red", "green"],
        ["VO", "GPS"],
    )


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
