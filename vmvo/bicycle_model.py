"""Bicycle model for vehicle motion control."""
import unittest

import numpy as np

from vmvo.constants import MAX_ACCEL, MAX_STEER, STEERING_RATIO, WHEEL_BASE
from vmvo.schema import State

STARTING_STATE = State(x=0.0, y=0.0, theta=0.0, velocity=0.0)


class BicycleModel:
    """Bicycle model for vehicle motion control."""
    def __init__(
        self,
        state: State = STARTING_STATE,
        wheel_base: float = WHEEL_BASE,
        steering_ratio: float = STEERING_RATIO,
        max_steer: float = MAX_STEER,
        max_accel: float = MAX_ACCEL,
    ) -> None:
        self.state = state
        self.wheel_base = wheel_base
        self.steering_ratio = steering_ratio
        self.max_steer = max_steer
        self.max_accel = max_accel

    def run(
        self,
        steering_angle: float,  # Degrees
        velocity: float,  # m/s
        dt: float,  # s
    ) -> State:
        """Run the bicycle model for one timestep"""
        # Check if steering angle is within bounds
        assert steering_angle <= self.max_steer, "Steering angle is too large"
        assert steering_angle >= -self.max_steer, "Steering angle is too small"

        estimated_accel = (velocity - self.state.velocity) / dt
        assert estimated_accel <= self.max_accel, "Acceleration is too large"

        x_next = State(**self.state.model_dump())
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

    def set_state(self, state: State) -> None:
        """Set the state of the bicycle model"""
        self.state = state


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
