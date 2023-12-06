"""
Models
"""
from typing import List

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
            self.time[key]
        )

    def __repr__(self):
        return f"Trajectory(len={len(self)})"

    def __str__(self):
        return self.__repr__()
