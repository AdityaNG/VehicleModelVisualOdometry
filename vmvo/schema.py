"""
Models
"""
from typing import List

import numpy as np
from pydantic import BaseModel


class State(BaseModel):
    """State of the vehicle"""

    x: float  # m
    y: float  # m
    theta: float  # radians
    velocity: float  # m/s


class Trajectory(BaseModel):
    """Trajectory of the vehicle"""

    x: List[float]  # m
    y: List[float]  # m
    theta: List[float]  # radians
    velocity: List[float]  # m/s
    time: List[float]  # s

    def __len__(self):
        return len(self.x)

    def __getitem__(self, key):
        return (
            self.x[key],
            self.y[key],
            self.theta[key],
            self.velocity[key],
            self.time[key],
        )

    def __repr__(self):
        return f"Trajectory(len={len(self)})"

    def __str__(self):
        return self.__repr__()

    def sub_trajectory_old(self, start: int, end: int):
        """Get a sub-trajectory from start to end"""
        x = self.x[start:end]
        y = self.y[start:end]
        theta = self.theta[start:end]

        x_homo = np.zeros((len(theta), 3, 3))
        x_homo[:, 0, 0] = np.cos(theta)
        x_homo[:, 0, 1] = -np.sin(theta)
        x_homo[:, 1, 0] = np.sin(theta)
        x_homo[:, 1, 1] = np.cos(theta)

        x_homo[:, 0, 2] = x
        x_homo[:, 1, 2] = y
        x_homo[:, 2, 2] = 1

        x_init = x_homo[0, :, :]

        # Rotate the trajectory so that the starting angle is 0
        x_homo = np.dot(x_homo, np.linalg.inv(x_init))
        # x_homo = np.dot(x_homo, x_init)

        x_new = x_homo[:, 0, 2]
        y_new = x_homo[:, 1, 2]
        theta_new = np.arccos(x_homo[:, 0, 0])

        return Trajectory(
            x=x_new,
            y=y_new,
            theta=theta_new,
            velocity=self.velocity[start:end],
            time=self.time[start:end],
        )

    def sub_trajectory(self, start: int, end: int):
        """Get a sub-trajectory from start to end"""
        x = np.array(self.x[start:end])
        y = np.array(self.y[start:end])
        theta = np.array(self.theta[start:end])

        theta_o = theta[0]

        rot = np.array(
            [
                [np.cos(theta_o), -np.sin(theta_o)],
                [np.sin(theta_o), np.cos(theta_o)],
            ]
        )

        x_homo = np.zeros((len(theta), 2))
        x_homo[:, 0] = x
        x_homo[:, 1] = y

        x_init = x_homo[0, :]

        x_homo = x_homo - x_init

        # Rotate the trajectory so that the starting angle is 0
        # x_homo = np.dot(x_homo, np.linalg.inv(rot))
        x_homo = np.dot(x_homo, rot)

        x_new = x_homo[:, 0]
        y_new = x_homo[:, 1]
        theta_new = np.array(theta) - theta_o

        return Trajectory(
            x=x_new,
            y=y_new,
            theta=theta_new,
            velocity=self.velocity[start:end],
            time=self.time[start:end],
        )

    def sub_trajectory_from_time(self, start_time: float, end_time: float):
        """Get a sub-trajectory from start to end based on time"""
        start_index = np.searchsorted(self.time, start_time, side="left")
        end_index = np.searchsorted(self.time, end_time, side="right")

        return self.sub_trajectory(
            start=start_index,
            end=end_index,
        )
