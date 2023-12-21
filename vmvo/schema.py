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

    steering_angle: float  # degrees


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

    def to_numpy(self):
        return np.array(
            [
                self.x,
                self.y,
                self.theta,
                self.velocity,
                self.time,
            ]
        ).T

    def sub_trajectory(self, start: int, end: int, theta_window: int = 10):
        """Get a sub-trajectory from start to end
        Transform the trajectory such that heading of the
            0th point is 0
        """
        x = np.array(self.x[start:end])
        y = np.array(self.y[start:end])
        theta = np.array(self.theta[start:end])

        theta_o = theta[0]

        # # Compute theta_o as the average of [-theta_window, theta_window]
        # theta_window_slice = np.array(self.theta[
        #     max(0, start - theta_window):
        #     min(len(self.theta), start + theta_window)
        # ])
        # # Remove NaNs
        # theta_window_slice = theta_window_slice[
        #     ~np.isnan(theta_window_slice)
        # ]
        # theta_window_slice = theta_window_slice[
        #     ~np.isinf(theta_window_slice)
        # ]
        # # Compute the mean of theta considering it ranges from 0 to 2pi
        # # Convert theta values to complex numbers and take the mean
        # theta_o = np.angle(np.mean(np.exp(1j * theta_window_slice)))

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

        assert end_index > start_index, "No frames found"

        return self.sub_trajectory(
            start=start_index,
            end=end_index,
        )


def states_list_to_trajectory(
    states: List[State],
    start_time: float,
    dt: float,
) -> Trajectory:
    timestamps = [start_time + i * dt for i in range(len(states))]
    x, y, theta, velocity = [], [], [], []

    for i in range(len(states)):
        state = states[i]
        x.append(state.x)
        y.append(state.y)
        theta.append(state.theta)
        velocity.append(state.velocity)

    return Trajectory(
        x=x, y=y, theta=theta, velocity=velocity, time=timestamps
    )


class GPTLabel(BaseModel):
    height: float
    width: float
    length: float
    X: float
    Y: float
    Z: float
    rot: float
    done: bool
    drop: bool


# class GPTOrientation(BaseModel):
#     _theta: float = np.pi / 2

#     @property
#     def theta(self):
#         return self._theta

#     @theta.setter
#     def theta(self, value):
#         self._theta = value % (2 * np.pi)


class GPTOrientation(BaseModel):
    # theta: float = np.pi / 2
    theta: float
