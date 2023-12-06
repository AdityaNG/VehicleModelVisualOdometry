"""Trajectory utilities"""
import math
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vmvo.schema import Trajectory


def process_vo_trajectory(
    trajectory: pd.DataFrame, scale: float = 1
) -> Trajectory:
    """Converts a trajectory dataframe to a Trajectory object
    trajectory['rot'] is an array of 3x3 rotation matrices
    Convert this to an array of yaw angles

    Velocity is not provided, so we will compute it using the distance between
    two points and the time between them
    """
    theta = []
    for rot in trajectory["rot"].tolist():
        theta.append(np.arctan2(rot[1, 0], rot[0, 0]))
    velocity = []
    for i in range(len(trajectory["x"]) - 1):
        dist = np.sqrt(
            (trajectory["x"][i] - trajectory["x"][i + 1]) ** 2
            + (trajectory["y"][i] - trajectory["y"][i + 1]) ** 2
        )
        time = trajectory["Timestamp"][i + 1] - trajectory["Timestamp"][i]
        velocity.append(dist / time)
    return Trajectory(
        x=np.array(trajectory["x"].tolist()) * scale,
        y=np.array(trajectory["y"].tolist()) * scale,
        theta=np.array(theta),
        velocity=np.array(velocity),
        time=np.array(trajectory["Timestamp"].tolist()),
    )


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
    trajectory: pd.DataFrame, num_frames: int = 25
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
    for i in range(num_frames):
        initial_heading += trajectory["heading"][i]
    initial_heading /= num_frames
    # Compute direction
    direction = []
    for heading in trajectory["heading"].tolist():
        direction.append(heading - initial_heading)

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

    return Trajectory(
        x=np.array(x),
        y=np.array(y),
        theta=np.array(direction),
        velocity=np.array(trajectory["speed"].tolist()),
        time=np.array(trajectory["Timestamp"].tolist()),
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
    plt.show()
