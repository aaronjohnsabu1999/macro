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

from macro.auction import auction_map
from macro.utils import time_to_true_anomaly
from macro.maneuvers import compute_state_transition_matrix, compute_maneuver_points
from macro.graph import SystemGraph
from macro.agent import Agent, OrbitalParams, SimulationParams

class GlideslopeSimulator:
    def __init__(
        self,
        agents: np.ndarray,
        neighbor_matrix: np.ndarray,
        neighbor_radius: float,
        target_positions: np.ndarray,
        orbit: OrbitalParams,
        sim: SimulationParams,
    ):
        self.agents = agents
        self.neighbors_matrix = neighbor_matrix
        self.neighbor_radius = neighbor_radius
        self.target_positions = target_positions

        self.orbital_radius = orbit.radius
        self.eccentricity = orbit.eccentricity
        self.angular_momentum = orbit.angular_momentum
        self.angular_velocity = orbit.angular_velocity

        self.set_simulation_params(sim)

    def set_simulation_params(self, sim: SimulationParams):
        self.timestep = sim.timestep
        self.num_jumps = sim.num_jumps
        self.num_frames = sim.num_frames
        self.auction_type = sim.auction_type
        self.layer_config = sim.layer_config
        self.flat_config = sim.flat_config

        self.steps_per_jump = int(self.num_frames / self.num_jumps)
        self.trajectory = np.zeros((len(self.agents), self.num_frames, 6))
        self.energy = 0.0
        self.deltavs = np.array([], dtype=float)
        self.all_system_graphs = []

    def run(self):
        for agent in self.agents:
            agent.update_direction()

        for jump in range(self.num_jumps):
            t0 = self.timestep * self.steps_per_jump * jump
            t1 = t0 + self.timestep * self.steps_per_jump
            phi = compute_state_transition_matrix(
                times=(t0, t1),
                eccentricity=self.eccentricity,
                orbital_radius=self.orbital_radius,
                angular_velocity=self.angular_velocity,
            )[:3]
            # phi_r = np.array([row[:3] for row in phi])
            # phi_v = np.array([row[3:6] for row in phi])
            phi_r = phi[:3, :3]
            phi_v = phi[:3, 3:6]

            system_graph = SystemGraph(
                layer_config=self.layer_config,
                flat_config=self.flat_config,
            )

            if self.auction_type:
                if self.auction_type not in auction_map:
                    raise ValueError(f"Unknown auction type: {self.auction_type}")
                if isinstance(self.target_positions, list):
                    raise ValueError(
                        "target_positions should be a numpy array, not a list."
                    )
                self.target_positions = auction_map[self.auction_type](
                    initial_positions=np.array([a.position for a in self.agents]),
                    target_positions=self.target_positions,
                    graph=system_graph,
                ).assign()
            else:
                self.target_positions = np.array([a.position for a in self.agents])
            for i, agent in enumerate(self.agents):
                agent.target = self.target_positions[i]
                agent.update_direction()

            for i, agent in enumerate(self.agents):
                deltav = agent.compute_delta_v(jump, self.num_jumps, phi_r, phi_v)
                self.energy += np.linalg.norm(deltav)
                self.deltavs = np.append(self.deltavs, deltav)

                for step in range(self.steps_per_jump):
                    r_t, v_t = compute_maneuver_points(
                        r_0=agent.position,
                        v_0=agent.velocity,
                        times=(t0, t0 + self.timestep * step),
                        eccentricity=self.eccentricity,
                        orbital_radius=self.orbital_radius,
                        angular_velocity=self.angular_velocity,
                    )
                    idx = jump * self.steps_per_jump + step
                    self.trajectory[agent.agent_id, idx] = np.concatenate((r_t, v_t))
                agent.position = r_t
                agent.velocity = v_t

            self.all_system_graphs.append(system_graph)

        return (
            self.trajectory,
            self.all_system_graphs,
            self.deltavs,
            self.energy,
        )

    def get_last_system_graph(self):
        """Return the last system graph after all jumps."""
        if self.all_system_graphs:
            return self.all_system_graphs[-1]
        return None
