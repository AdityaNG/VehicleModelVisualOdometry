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
