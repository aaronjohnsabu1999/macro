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
from scipy.optimize import fsolve

from macro.utils import to_layer_index


def calc_number_of_layers(num_agents):
    """
    Determine the number of concentric layers and their agent counts
    such that the total equals or exceeds the given number of agents.

    Parameters
    ----------
    num_agents : int
        Total number of agents to distribute in hexagonal layers.

    Returns
    -------
    tuple[int, list[int]]
        Number of layers and list of agent counts per layer.
    """
    layer_lengths = []
    layer = 1
    total = 0

    while total < num_agents:
        count = layer * 6
        layer_lengths.append(count)
        total += count
        layer += 1

    return len(layer_lengths), layer_lengths


def generate_target_positions(
    clear_aperture: float,
    orbital_radius: float,
    num_agents: int,
    layer_lengths: list[int],
    expansion: float,
):
    """
    Generate 3D target positions for agents arranged in concentric circular layers.

    Parameters
    ----------
    clear_aperture : float
        Aperture diameter of the telescope system.
    orbital_radius : float
        Reference orbital radius.
    num_agents : int
        Total number of agents to be positioned.
    layer_lengths : list[int]
        Number of agents in each layer.
    expansion : float
        Radial and axial scaling factor.

    Returns
    -------
    np.ndarray
        3D positions of agents in the format [[x1, y1, z1], [x2, y2, z2], ...].
    """

    R_f = np.zeros((num_agents, 3), dtype=float)
    num_layers = len(layer_lengths)
    _, layer_params = generate_paraboloid_layer_geometry(
        clear_aperture, orbital_radius, num_layers
    )

    XY_radii = [r[0] for r in layer_params]
    Z_offsets = [r[1] for r in layer_params]

    for agent_id in range(num_agents):
        layer_idx, pos_idx = to_layer_index(layer_lengths, agent_id)
        angle = 2 * np.pi * pos_idx / layer_lengths[layer_idx]

        x = expansion * XY_radii[layer_idx] * np.sin(angle)
        y = expansion * XY_radii[layer_idx] * np.cos(angle)
        z = expansion * Z_offsets[layer_idx]

        R_f[agent_id] = [x, y, z]

    return R_f


def compute_desired_attitudes(
    clear_aperture, orbital_radius, layer_lengths, flat_config
):
    """
    Calculate the desired attitude (roll, pitch, yaw) for each agent based on their target position.

    Parameters
    ----------
    clear_aperture : float
        Aperture diameter of the telescope system.
    orbital_radius : float
        Reference orbital radius.
    layer_lengths : list[int]
        Number of agents in each concentric layer.
    flat_config : list[int]
        Flattened ordering of agent IDs (i.e., final config as flat list).

    Returns
    -------
    list[list[float]]
        A list of attitude vectors [roll, pitch, yaw] for each agent.
    """
    num_agents = len(flat_config)
    desired_attitudes = np.zeros((num_agents + 1, 3), dtype=float)
    pitch_angles, _ = generate_paraboloid_layer_geometry(
        clear_aperture, orbital_radius, len(layer_lengths)
    )

    for flat_idx, agent_id in enumerate(flat_config):
        pres_layer, pres_pos = to_layer_index(layer_lengths, flat_idx)
        true_layer, _ = to_layer_index(layer_lengths, agent_id)

        # Roll is 0.0 by default
        desired_attitudes[agent_id, 1] = np.pi / 2.0 - pitch_angles[true_layer]  # Pitch
        desired_attitudes[agent_id, 2] = (2.0 * np.pi * pres_pos) / layer_lengths[
            pres_layer
        ]  # Yaw

        # Normalize yaw to [-π, π]
        if desired_attitudes[agent_id, 2] > np.pi:
            desired_attitudes[agent_id, 2] -= 2.0 * np.pi

    return desired_attitudes


def generate_paraboloid_layer_geometry(clear_aperture, orbital_radius, num_layers):
    """
    Generate concentric layer parameters for agents arranged on a parabolic surface.

    Parameters
    ----------
    clear_aperture : float
        Diameter of the telescope aperture.
    orbital_radius : float
        Distance from the center to the vertex of the parabola.
    num_layers : int
        Number of concentric layers to compute.

    Returns
    -------
    tuple
        - List of pitch angles (Thetas) for each layer
        - List of (XY_radius, Z_offset) tuples for each layer
    """

    def surface_equation(theta_i, theta_list):
        sin_sum = np.sum(np.sin(theta_list)) + np.sin(theta_i)
        cos_sum = np.sum(np.cos(theta_list)) + np.cos(theta_i)
        lhs = clear_aperture * (0.5 + sin_sum)
        rhs = clear_aperture * (clear_aperture / (8.0 * orbital_radius) + cos_sum)
        return (lhs**2) / (2.0 * orbital_radius) - rhs

    thetas = []
    positions = [(clear_aperture / 2.0, (clear_aperture**2) / (8.0 * orbital_radius))]

    for _ in range(num_layers):
        theta_i = fsolve(lambda t: surface_equation(t, thetas), 0)[0]
        thetas.append(theta_i)

        xy_radius = clear_aperture * (0.5 + np.sum(np.sin(thetas)))
        z_offset = clear_aperture * (
            clear_aperture / (8.0 * orbital_radius) + np.sum(np.cos(thetas))
        )
        positions.append((xy_radius, z_offset))

    return thetas, positions
