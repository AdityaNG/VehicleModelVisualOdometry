"""
MPC for the bicycle model

TODO: Test this code
"""

import numpy as np
from scipy.optimize import minimize

from vmvo.bicycle_model import BicycleModel
from vmvo.schema import State, Trajectory


def mpc_run(
    trajectory: Trajectory,
    bicycle_model: BicycleModel,
    velocity: float,
    starting_steering_angle: float,
    time_step: float,
):
    """
    Run MPC to optimize for a given cost function

    K is the inverse agressiveness of the steering input
    Smaller K values correspond to more aggressive steering
    Larger K values correspond to less aggressive steering
    """
    # tuning parameter for the cost function
    # K = 0.000005
    # K = 0.00000000000000000000005
    K = 0.0
    # K = 0.00000000000005
    # K = 0.0005

    trajectory_np = trajectory.to_numpy()[:, [0, 1]]

    dt = time_step
    trajectory_interp = traverse_trajectory(trajectory_np, velocity * dt)

    N = len(trajectory_interp) - 1

    if trajectory_interp.shape[0] <= 1:
        return np.zeros(N)

    start_state = State(
        x=0.0,
        y=0.0,
        theta=0.0,
        velocity=velocity,
        steering_angle=starting_steering_angle,
    )

    bicycle_model.set_state(start_state)

    # define the model
    def bicycle_model_function(x, u):
        start_state = State(
            x=x[0], y=x[1], theta=x[2], velocity=x[3], steering_angle=u
        )
        bicycle_model.set_state(start_state)
        next_state = bicycle_model.run(u, x[3], dt)
        x_next = np.array(
            [next_state.x, next_state.y, next_state.theta, next_state.velocity]
        )
        return x_next

    # define the cost function
    def cost(u, x, x_des):
        cost_val = 0.0
        for i in range(N):
            x = bicycle_model_function(x, u[i])

            # Compute cost for each point on the trajectory
            cost_val += (
                (x[0] - x_des[i, 0]) ** 2
                + (x[1] - x_des[i, 1]) ** 2
                + K * u[i] ** 2
            )

        return cost_val

    # initial state and input sequence
    x0 = np.array(
        [trajectory_interp[0, 0], trajectory_interp[0, 1], 0.0, velocity]
    )
    # x0 = np.array([0.0, 0.0, 0.0, velocity])
    u0 = np.zeros(N)

    # bounds on the steering angle
    bounds = []
    for _ in range(N):
        bounds.append(
            (
                max(
                    [
                        # starting_steering_angle -
                        # bicycle_model.max_steer_rate * dt * i,
                        -bicycle_model.max_steer,
                    ]
                ),
                min(
                    [
                        # starting_steering_angle +
                        # bicycle_model.max_steer_rate * dt * i,
                        bicycle_model.max_steer,
                    ]
                ),
            )
        )

    # optimize the cost function
    res = minimize(
        cost,
        u0,
        args=(x0, trajectory_interp.copy()),
        method="SLSQP",
        bounds=bounds,
        options=dict(maxiter=100),
    )

    u_opt = res.x
    return u_opt


def traverse_trajectory(traj, D):
    traj_interp = [traj[0]]
    dist = 0.0
    total_dist = 0.0
    for traj_i in range(1, traj.shape[0]):
        traj_dist = (
            (traj[traj_i, 0] - traj[traj_i - 1, 0]) ** 2
            + (traj[traj_i, 1] - traj[traj_i - 1, 1]) ** 2
        ) ** 0.5
        if dist + traj_dist > D:
            traj_interp.append(traj[traj_i - 1])
            dist = traj_dist
        else:
            dist += traj_dist
        total_dist += traj_dist

    return np.array(traj_interp)
