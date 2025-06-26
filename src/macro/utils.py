# /***********************************************************
# *                                                         *
# * Copyright (c) 2025                                      *
# *                                                         *
# * Indian Institute of Technology, Bombay                  *
# *                                                         *
# * Author(s): Aaron John Sabu, Dwaipayan Mukherjee         *
# * Contact  : aaronjs@g.ucla.edu, dm@ee.iitb.ac.in         *
# *                                                         *
# ***********************************************************/

import numpy as np


def random_sign():
    """Return either +1 or -1 randomly."""
    return 1 if np.random.randint(0, 2) == 0 else -1


def random_vector(scale=1.0):
    """Generate a 3D vector with random direction and uniform magnitude up to `scale`."""
    return scale * np.array([random_sign() * np.random.rand() for _ in range(3)])


def init_pose(num_agents, radius, *, seed=None):
    """
    Generate random initial positions and attitudes for spacecraft agents.

    Parameters:
    ----------
    num_agents : int
      Number of spacecraft agents
    radius : float
      Maximum position magnitude from the origin
    seed : int, optional
      Random seed for reproducibility

    Returns:
    -------
    R_0 : list of np.ndarray
      Initial position vectors for each agent
    Theta : list of np.ndarray
      Initial attitude vectors (Euler angles in radians)
    """
    if seed is not None:
        np.random.seed(seed)

    R_0 = [random_vector(radius) for _ in range(num_agents)]
    Theta = [random_vector(np.pi) for _ in range(num_agents)]
    Theta.append(np.zeros(3))

    return R_0, Theta


def time_to_true_anomaly(
    ang_vel: float | np.ndarray,
    eccentricity: float | np.ndarray,
    time_since_periapsis: float | np.ndarray,
    calc_derivative: bool = False,
) -> tuple[np.ndarray, ...]:
    """
    Compute true anomaly (and optionally its derivatives) from time since periapsis
    using an approximation to Kepler's Equation.

    Parameters:
    ----------
    ang_vel : float or np.ndarray
      Mean motion (rad/s)
    eccentricity : float or np.ndarray
      Orbital eccentricity
    time_since_periapsis : float or np.ndarray
      Time since periapsis passage (s)
    calc_derivative : bool, default=False
      If True, return first and second time derivatives of the anomaly

    Returns:
    -------
    F : np.ndarray
      True anomaly (radians)
    F_dot : np.ndarray (optional)
      First derivative of F with respect to time
    F_ddot : np.ndarray (optional)
      Second derivative of F with respect to time
    """

    ang_vel = np.asarray(ang_vel, dtype=float)
    eccentricity = np.asarray(eccentricity, dtype=float)
    time_since_periapsis = np.asarray(time_since_periapsis, dtype=float)

    # Mean anomaly
    mean_anomaly = ang_vel * time_since_periapsis

    # Fourier expansion approximation
    F = (
        mean_anomaly
        + (2.0 * eccentricity - (eccentricity**3.0) / 4.0) * np.sin(mean_anomaly)
        + (5.0 / 4) * (eccentricity**2.0) * np.sin(2.0 * mean_anomaly)
        + (13.0 / 12) * (eccentricity**3.0) * np.sin(3.0 * mean_anomaly)
    )
    if not calc_derivative:
        return (F,)

    # Derivatives of true anomaly
    F_dot = (
        (1.0 - (eccentricity**2.0)) ** (3.0 / 2)
        / (1.0 + (eccentricity * np.cos(F))) ** 2
    ) * ang_vel
    F_ddot = (
        -2.0
        * eccentricity
        * (F_dot**2.0)
        * np.sin(F)
        / (1 + (eccentricity * np.cos(F)))
    )

    return F, F_dot, F_ddot


def consensus_step(
    agent_index: int,
    attitude_history: np.ndarray,
    desired_attitude: np.ndarray,
    neighbor_indices: np.ndarray,
    control_gains: np.ndarray,
    step_size: float,
    iteration_count: int,
):
    """
    Perform one attitude update step for a given agent using consensus-based control.

    Parameters:
      agent_index (int): Index of the ego agent performing the update.
      attitude_history (np.ndarray): Time history of attitude vectors for each agent.
      desired_attitude (np.ndarray): Desired attitude vectors for each agent.
      neighbor_indices (np.ndarray): Indices of neighbors of the ego agent.
      control_gains (np.ndarray): Per-agent control gain vectors (one per agent).
      step_size (float): Base step size (scalar multiplier for update magnitude).
      iteration_count (int): Current iteration (used to scale the update).

    Returns:
      np.ndarray: Updated attitude vector for the ego agent.
    """
    previous_attitude = attitude_history[agent_index][-1]
    updated_attitude = previous_attitude.copy()

    for neighbor_index, gain_vector in enumerate(control_gains):
        if neighbor_index in neighbor_indices:
            ego_error = (
                attitude_history[agent_index][-1] - desired_attitude[agent_index]
            )
            neighbor_error = (
                attitude_history[neighbor_index][-1] - desired_attitude[neighbor_index]
            )
            error_difference = ego_error - neighbor_error

            scaled_adjustment = (
                step_size * iteration_count * gain_vector * error_difference
            )
            updated_attitude = updated_attitude - scaled_adjustment

    return updated_attitude


class NullLogger:
    """
    A logger that does nothing. Used to disable logging in certain contexts.
    """

    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def critical(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def setLevel(self, level):
        pass

    def addHandler(self, handler):
        pass

    def removeHandler(self, handler):
        pass

    def handlers(self):
        return []

    def hasHandlers(self):
        return False

    def getEffectiveLevel(self):
        return 0
