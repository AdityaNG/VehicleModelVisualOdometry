"""Trajectory utilities"""
import math
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vmvo.schema import Trajectory


def process_vo_trajectory(
    trajectory: pd.DataFrame, scale: float = 0.25
) -> Trajectory:
    """Converts a trajectory dataframe to a Trajectory object
    trajectory['rot'] is an array of 3x3 rotation matrices
    Convert this to an array of yaw angles

    Velocity is not provided, so we will compute it using the distance between
    two points and the time between them
    """
    ax = "x"
    ay = "y"
    theta = []
    for rot in trajectory["rot"].tolist():
        theta.append(np.arctan2(rot[1, 0], rot[0, 0]))
        # theta.append(np.arctan2(rot[2, 1], rot[2, 2]))
        # theta.append(
        #     np.arctan2(-rot[2, 0], (rot[2, 2]**2 + rot[2, 2]**2)**0.5)
        # )
    velocity = [
        0,
    ]
    for i in range(len(trajectory[ax]) - 1):
        dist = np.sqrt(
            (trajectory[ax][i] - trajectory[ax][i + 1]) ** 2
            + (trajectory[ay][i] - trajectory[ay][i + 1]) ** 2
        )
        time = trajectory["Timestamp"][i + 1] - trajectory["Timestamp"][i]
        velocity.append(dist / time)

    traj = np.zeros(
        (
            len(trajectory[ax]),
            2,
        )
    )
    traj[:, 0] = trajectory[ax]
    traj[:, 1] = trajectory[ay]

    traj = smoothen_traj(traj, window_size=20)

    x_list = traj[:, 0]
    y_list = traj[:, 1]

    return Trajectory(
        x=np.array(x_list) * scale,
        y=np.array(y_list) * scale,
        theta=np.array(theta),
        velocity=np.array(velocity),
        time=np.array(trajectory["Timestamp"].tolist()) / 1000.0,
    )


def smoothen_traj(trajectory, window_size=3):
    """
    Smoothen a trajectory using moving average.

    Args:
    trajectory (list): List of 3D points [(x1, y1, z1), (x2, y2, z2), ...].
        window_size (int): Size of the moving average window.

    Returns:
        list: Smoothened trajectory as a list of 3D points.
    """
    smoothed_traj = []
    num_points = len(trajectory)

    # Handle edge cases
    if num_points <= window_size:
        return trajectory

    # Calculate the moving average for each point
    for i in range(num_points):
        window_start = max(0, i - window_size + 1)
        window_end = min(i + 1, num_points)
        window_points = trajectory[window_start:window_end]
        avg_point = (
            sum(p[0] for p in window_points) / len(window_points),
            sum(p[1] for p in window_points) / len(window_points),
        )
        smoothed_traj.append(avg_point)

    smoothed_traj = np.array(smoothed_traj)

    return smoothed_traj


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


def geodetic_to_euclidean(p1, p2):  # pylint: disable=R0914
    """Converts geodetic coordinates to euclidean coordinates"""
    # WGS84 ellipsoid constants
    a = 6378137  # Earth's radius in meters
    e = 8.1819190842622e-2  # Eccentricity

    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])

    # Calculate X, Y, Z coordinates
    x1 = (
        a
        / math.sqrt(1 - e**2 * math.sin(lat1) ** 2)
        * math.cos(lat1)
        * math.cos(lon1)
    )
    y1 = (
        a
        / math.sqrt(1 - e**2 * math.sin(lat1) ** 2)
        * math.cos(lat1)
        * math.sin(lon1)
    )
    z1 = (
        a
        * (1 - e**2)
        / math.sqrt(1 - e**2 * math.sin(lat1) ** 2)
        * math.sin(lat1)
    )

    x2 = (
        a
        / math.sqrt(1 - e**2 * math.sin(lat2) ** 2)
        * math.cos(lat2)
        * math.cos(lon2)
    )
    y2 = (
        a
        / math.sqrt(1 - e**2 * math.sin(lat2) ** 2)
        * math.cos(lat2)
        * math.sin(lon2)
    )
    z2 = (
        a
        * (1 - e**2)
        / math.sqrt(1 - e**2 * math.sin(lat2) ** 2)
        * math.sin(lat2)
    )

    x = x2 - x1
    y = y2 - y1
    z = z2 - z1

    return (x, y, z)


def process_gps_trajectory(
    trajectory: pd.DataFrame, heading_num_frames: int = 25
) -> Trajectory:
    """Converts a trajectory dataframe to a Trajectory object
    For direction, we have a heading column in degrees
    Use the initial N frames to compute the initial heading
    Then, use the heading column to compute the direction

    Convert the Latitude and Longitude to x and y coordinates
    Use the first point as the origin
    Apply the Haversine formula to compute the distance between two points
    """
    # Compute initial heading
    initial_heading = 0.0
    for i in range(heading_num_frames):
        initial_heading += trajectory["heading"][i]
    initial_heading /= heading_num_frames
    # Compute direction
    direction = []
    for heading in trajectory["heading"].tolist():
        direction.append(heading - initial_heading)

    direction = np.radians(direction)
    # Convert to x and y coordinates
    # Use the first point as the origin
    # Apply the Haversine formula to compute the distance between two points
    lat1 = trajectory["Latitude"][0]
    lon1 = trajectory["Longitude"][0]
    x = [0]
    y = [0]
    for i in range(len(trajectory["Latitude"])):
        lat2 = trajectory["Latitude"][i]
        lon2 = trajectory["Longitude"][i]

        dx, dy, _ = geodetic_to_euclidean((lat1, lon1), (lat2, lon2))

        x.append(dx + x[-1])
        y.append(dy + y[-1])
        lat1 = lat2
        lon1 = lon2

    # GPS updates at 10 Hz
    # But data logging happens at 20 Hz
    # Hence many frames have repeated data
    # Remove the repeated data and interpolate
    x = np.array(x)
    y = np.array(y)
    direction = np.array(direction)
    velocity = np.array(trajectory["speed"].tolist())
    time = np.array(trajectory["Timestamp"].tolist()) / 1000.0

    # Replace repeated data with interpolation
    x_new = []
    y_new = []
    direction_new = []
    velocity_new = []
    time_new = []
    last_update = 0

    x_new.append(x[last_update])
    y_new.append(y[last_update])
    direction_new.append(direction[last_update])
    velocity_new.append(velocity[last_update])
    time_new.append(time[last_update])

    for i in range(1, len(x)):
        if x[last_update] != x[i] or y[last_update] != y[i]:
            # Interpolate between last_update and i
            for j in range(last_update + 1, i + 1):
                alpha = (j - last_update) / (i - last_update)
                x_new.append(x[last_update] * (1 - alpha) + x[i] * alpha)
                y_new.append(y[last_update] * (1 - alpha) + y[i] * alpha)
                direction_new.append(
                    direction[last_update] * (1 - alpha) + direction[i] * alpha
                )
                velocity_new.append(
                    velocity[last_update] * (1 - alpha) + velocity[i] * alpha
                )
                time_new.append(
                    time[last_update] * (1 - alpha) + time[i] * alpha
                )

            last_update = i

    # Interpolate the last few points
    for j in range(last_update + 1, len(x)):
        alpha = (j - last_update) / (len(x) - last_update)
        x_new.append(x[last_update] * (1 - alpha) + x[-1] * alpha)
        y_new.append(y[last_update] * (1 - alpha) + y[-1] * alpha)
        direction_new.append(
            direction[last_update] * (1 - alpha) + direction[-1] * alpha
        )
        velocity_new.append(
            velocity[last_update] * (1 - alpha) + velocity[-1] * alpha
        )
        time_new.append(time[last_update] * (1 - alpha) + time[-1] * alpha)

    # Length of the new trajectory must be same
    # as the length of the original trajectory
    assert len(x_new) == len(x), f"Length mismatch {len(x_new)} != {len(x)}"

    return Trajectory(
        x=-np.array(x_new),
        y=np.array(y_new),
        theta=np.array(direction_new),
        velocity=np.array(velocity_new),
        time=np.array(time_new),
    )


def visualize_trajectory(
    trajectory: Trajectory,
    dim: int = 500,
    color: tuple = (0, 255, 0),
) -> np.ndarray:
    """Visualize the trajectory
    Draw the trajectory using opencv
    """
    res = np.zeros((dim, dim, 3), dtype=np.uint8)
    x_norm = (trajectory.x - np.min(trajectory.x)) / (
        np.max(trajectory.x) - np.min(trajectory.x)
    )
    y_norm = (trajectory.y - np.min(trajectory.y)) / (
        np.max(trajectory.y) - np.min(trajectory.y)
    )
    # Draw the trajectory
    for i in range(len(trajectory.x) - 1):
        cv2.line(
            res,
            (int(x_norm[i] * dim), int(y_norm[i] * dim)),
            (int(x_norm[i + 1] * dim), int(y_norm[i + 1] * dim)),
            color,
            1,
        )
    return res


def plot_trajectory_list(
    trajectory_list: List[Trajectory],
    color_list: List[tuple],
    legend_list: List[str],
):
    """Plot a list of trajectories"""
    _, ax = plt.subplots()
    for trajectory, color in zip(trajectory_list, color_list):
        ax.plot(trajectory.x, trajectory.y, color=color)
    ax.legend(legend_list)
    ax.set_aspect("equal", "box")

    # Make sure plot is square
    x_min = min([min(trajectory.x) for trajectory in trajectory_list])
    x_max = max([max(trajectory.x) for trajectory in trajectory_list])
    y_min = min([min(trajectory.y) for trajectory in trajectory_list])
    y_max = max([max(trajectory.y) for trajectory in trajectory_list])

    x_range = x_max - x_min
    y_range = y_max - y_min

    if x_range > y_range:
        y_mid = (y_max + y_min) / 2
        y_min = y_mid - x_range / 2
        y_max = y_mid + x_range / 2
    else:
        x_mid = (x_max + x_min) / 2
        x_min = x_mid - y_range / 2
        x_max = x_mid + y_range / 2

    x_buf = 0.1 * (x_max - x_min)
    y_buf = 0.1 * (y_max - y_min)

    x_min -= x_buf
    x_max += x_buf
    y_min -= y_buf
    y_max += y_buf

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.show()


def convert_3D_points_to_2D(points_3D, homo_cam_mat):
    points_2D = []
    for index in range(points_3D.shape[0]):
        p4d = points_3D[index]
        p2d = (homo_cam_mat) @ p4d
        px, py = 0, 0
        if p2d[2][0] != 0.0:
            px, py = int(p2d[0][0] / p2d[2][0]), int(p2d[1][0] / p2d[2][0])

        points_2D.append([px, py])

    return np.array(points_2D)


def get_rect_coords(x_i, y_i, x_j, y_j, width=2.83972):
    Pi = np.array([x_i, y_i])
    Pj = np.array([x_j, y_j])
    height = np.linalg.norm(Pi - Pj)
    diagonal = (width**2 + height**2) ** 0.5
    D = diagonal / 2.0

    M = ((Pi + Pj) / 2.0).reshape((2,))
    theta = np.arctan2(Pi[1] - Pj[1], Pi[0] - Pj[0])
    theta += np.pi / 4.0
    points = np.array(
        [
            M
            + np.array(
                [
                    D * np.sin(theta + 0 * np.pi / 2.0),
                    D * np.cos(theta + 0 * np.pi / 2.0),
                ]
            ),
            M
            + np.array(
                [
                    D * np.sin(theta + 1 * np.pi / 2.0),
                    D * np.cos(theta + 1 * np.pi / 2.0),
                ]
            ),
            M
            + np.array(
                [
                    D * np.sin(theta + 2 * np.pi / 2.0),
                    D * np.cos(theta + 2 * np.pi / 2.0),
                ]
            ),
            M
            + np.array(
                [
                    D * np.sin(theta + 3 * np.pi / 2.0),
                    D * np.cos(theta + 3 * np.pi / 2.0),
                ]
            ),
        ]
    )
    return points


def get_rect_coords_3D(Pi, Pj, width=2.83972):
    x_i, y_i = Pi[0, 0], Pi[2, 0]
    x_j, y_j = Pj[0, 0], Pj[2, 0]
    points_2D = get_rect_coords(x_i, y_i, x_j, y_j, width)
    points_3D = []
    for index in range(points_2D.shape[0]):
        # point_2D = points_2D[index]
        point_3D = Pi.copy()
        point_3D[0, 0] = points_2D[index, 0]
        point_3D[2, 0] = points_2D[index, 1]

        points_3D.append(point_3D)

    return np.array(points_3D)


def plot_steering_traj(  # pylint: disable=dangerous-default-value
    frame_center: np.ndarray,
    trajectory: Trajectory,
    shape=(1080, 1920),
    color=(255, 0, 0),
    intrinsic_matrix=None,
    DistCoef=None,
    offsets=(0, -5, -10),
    method="add_weighted",
):
    assert method in ("overlay", "mask", "add_weighted")

    # Save frame_center shape
    frame_center_shape = frame_center.shape[:2]
    # Resize frame_center to shape
    frame_center = cv2.resize(
        frame_center, shape[::-1], interpolation=cv2.INTER_AREA
    )

    if intrinsic_matrix is None:
        intrinsic_matrix = np.array(
            [
                [1250.6, 0, 978.4],
                [0, 1254.8, 562.1],
                [0, 0, 1.0],
            ]
        )
    if DistCoef is None:
        DistCoef = np.array(
            [
                0.0936,  # k1
                -0.5403,  # k2
                7.2525e-04,  # p1
                0.0084,  # p2
                0.7632,  # k3
            ]
        )

    homo_cam_mat = np.hstack((intrinsic_matrix, np.zeros((3, 1))))

    prev_point = None
    prev_point_3D = None
    rect_frame = np.zeros_like(frame_center)

    for trajectory_index in range(len(trajectory)):
        x = trajectory.x[trajectory_index]
        z = -trajectory.y[trajectory_index]
        y = np.zeros_like(z)
        p4d = np.ones((4, 1))
        p3d = np.array(
            [
                x * 1 - offsets[0],
                y * 1 - offsets[1],
                z * 1 - offsets[2],
            ]
        ).reshape((3, 1))
        p4d[:3, :] = p3d

        p2d = (homo_cam_mat) @ p4d
        if (
            p2d[2][0] != 0.0
            and not np.isnan(p2d).any()
            and not np.isinf(p2d).any()
        ):
            px, py = int(p2d[0][0] / p2d[2][0]), int(p2d[1][0] / p2d[2][0])
            if prev_point is not None:
                px_p, py_p = prev_point
                dist = ((px_p - px) ** 2 + (py_p - py) ** 2) ** 0.5
                if dist < 20:
                    rect_coords_3D = get_rect_coords_3D(p4d, prev_point_3D)
                    rect_coords = convert_3D_points_to_2D(
                        rect_coords_3D, homo_cam_mat
                    )
                    rect_frame = cv2.fillPoly(
                        rect_frame, pts=[rect_coords], color=color
                    )

                    frame_center = cv2.line(
                        frame_center, (px_p, py_p), (px, py), color, 2
                    )

            prev_point = (px, py)
            prev_point_3D = p4d.copy()
        else:
            prev_point = None
            prev_point_3D = None

    if method == "mask":
        mask = np.logical_and(
            rect_frame[:, :, 0] == color[0],
            rect_frame[:, :, 1] == color[1],
            rect_frame[:, :, 2] == color[2],
        )
        frame_center[mask] = color
    elif method == "overlay":
        frame_center += (0.5 * rect_frame).astype(np.uint8)
    elif method == "add_weighted":
        cv2.addWeighted(frame_center, 0.8, rect_frame, 0.2, 0.0, frame_center)

    # Resize frame_center back to frame_center_shape
    frame_center = cv2.resize(
        frame_center, frame_center_shape[::-1], interpolation=cv2.INTER_AREA
    )

    return frame_center


def plot_bev_trajectory(
    frame_center: np.ndarray,
    trajectory: Trajectory,
    color=(0, 255, 0),
    thickness=10,
):
    WIDTH, HEIGHT = frame_center.shape[1], frame_center.shape[0]
    traj_plot = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255

    X = np.array(trajectory.x)
    Z = -np.array(trajectory.y)

    print("x", X.min(), X.max())
    print("z", Z.min(), Z.max())

    X_min, X_max = -20.0, 20.0
    Z_min, Z_max = -20.0, 20.0
    X = (X - X_min) / (X_max - X_min)
    Z = (Z - Z_min) / (Z_max - Z_min)

    for traj_index in range(1, X.shape[0]):
        u = round(X[traj_index] * (WIDTH - 1))
        v = round(Z[traj_index] * (HEIGHT - 1))

        u_p = round(X[traj_index - 1] * (WIDTH - 1))
        v_p = round(Z[traj_index - 1] * (HEIGHT - 1))

        traj_plot = cv2.circle(traj_plot, (u, v), thickness, color, -1)
        traj_plot = cv2.line(traj_plot, (u_p, v_p), (u, v), color, thickness)

    traj_plot = cv2.flip(traj_plot, 0)
    return traj_plot
