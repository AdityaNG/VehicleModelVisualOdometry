"""Bicycle model for vehicle motion control."""
import unittest
from typing import List

import numpy as np

from vmvo.constants import (
    MAX_ACCEL,
    MAX_STEER,
    MAX_STEER_RATE,
    STEERING_RATIO,
    WHEEL_BASE,
)
from vmvo.schema import State

STARTING_STATE = State(
    x=0.0, y=0.0, theta=0.0, velocity=0.0, steering_angle=0.0
)


class BicycleModel:
    """Bicycle model for vehicle motion control."""

    def __init__(
        self,
        state: State = STARTING_STATE,
        wheel_base: float = WHEEL_BASE,
        steering_ratio: float = STEERING_RATIO,
        max_steer: float = MAX_STEER,
        max_steer_rate: float = MAX_STEER_RATE,
        max_accel: float = MAX_ACCEL,
    ) -> None:
        self.state = state
        self.wheel_base = wheel_base
        self.steering_ratio = steering_ratio
        self.max_steer = max_steer
        self.max_accel = max_accel
        self.max_steer_rate = max_steer_rate

    def run(
        self,
        steering_angle: float,  # Degrees
        velocity: float,  # m/s
        dt: float,  # s
    ) -> State:
        """Run the bicycle model for one timestep"""
        # Check if steering angle is within bounds
        assert (
            abs(steering_angle) <= self.max_steer
        ), "Steering angle is out of bounds"

        # Check if steering rate is within bounds
        # steering_rate = (steering_angle - self.state.steering_angle) / dt
        # assert (
        #     abs(steering_rate) <= self.max_steer_rate
        # ), f"Steering rate is out of bounds, {abs(steering_rate)}"
        # f" <= {self.max_steer_rate}"

        estimated_accel = (velocity - self.state.velocity) / dt
        assert (
            abs(estimated_accel) <= self.max_accel
        ), "Acceleration is out of bounds"

        x_next = State(**self.state.model_dump())
        x_next.steering_angle = steering_angle
        delta = np.radians(steering_angle) / STEERING_RATIO
        x_next.theta = self.state.theta + (
            velocity / WHEEL_BASE * np.tan(delta) * dt
        )  # theta
        x_next.x = self.state.x + (
            velocity * np.cos(x_next.theta) * dt
        )  # x pos
        x_next.y = self.state.y + (
            velocity * np.sin(x_next.theta) * dt
        )  # y pos
        x_next.velocity = velocity
        self.set_state(x_next)
        return x_next

    def run_sequence(
        self,
        steering_angles: np.ndarray,  # Degrees
        velocities: np.ndarray,  # m/s
        dt: float,  # s
    ) -> List[State]:
        """Run the bicycle model for a sequence of timesteps"""
        assert len(steering_angles) == len(velocities)
        states = []
        for steering_angle, velocity in zip(steering_angles, velocities):
            state = self.run(steering_angle, velocity, dt)
            states.append(state)
        return states

    def set_state(self, state: State) -> None:
        """Set the state of the bicycle model"""
        self.state = state

    def reset(self) -> None:
        """Reset the bicycle model"""
        self.state = STARTING_STATE


class TestBicycleModel(unittest.TestCase):
    """Test the bicycle model"""

    def setUp(self):
        """Set up the bicycle model"""
        self.bicycle = BicycleModel()

    def test_run(self):
        """Test the run function"""
        old_state = State(**self.bicycle.state.model_dump())
        state = self.bicycle.run(steering_angle=30.0, velocity=0.0, dt=0.1)
        self.assertEqual(state.x, old_state.x)
        self.assertEqual(state.y, old_state.y)
        self.assertEqual(state.theta, old_state.theta)
        self.assertEqual(state.velocity, old_state.velocity)


if __name__ == "__main__":
    unittest.main()
