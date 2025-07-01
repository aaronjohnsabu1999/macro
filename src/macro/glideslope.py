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
from dataclasses import dataclass
from macro.auction import auction_map
from macro.utils import time_to_true_anomaly
from macro.maneuvers import compute_state_transition_matrix, compute_maneuver_points
from macro.neighbors import neighborsConfig, neighborsDistance


@dataclass
class Agent:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    target: np.ndarray
    glideslope_dir: np.ndarray = None

    def update_direction(self):
        delta = self.position - self.target
        norm = np.linalg.norm(delta)
        self.glideslope_dir = delta / norm if norm > 1e-4 else np.zeros_like(delta)

    def compute_delta_v(
        self, idx: int, total_steps: int, phi_r: np.ndarray, phi_v: np.ndarray
    ) -> np.ndarray:
        if self.glideslope_dir is None or np.linalg.norm(self.glideslope_dir) < 1e-4:
            return np.zeros(3)

        scale = 1.0 - (idx + 1) / total_steps
        offset = (
            scale * np.linalg.norm(self.position - self.target) * self.glideslope_dir
        )
        r_next = self.target + offset

        delta_v = (
            np.linalg.inv(phi_v) @ (r_next - phi_r @ self.position) - self.velocity
        )
        self.velocity += delta_v
        return delta_v


@dataclass
class OrbitalParams:
    radius: float
    eccentricity: float
    angular_momentum: float
    angular_velocity: float


@dataclass
class SimulationParams:
    num_phases: int
    timestep: float
    total_frames: int
    auction_type: str = "Distributed"
    use_config: bool = False
    config_data: tuple = ()
    config_order: tuple = ()


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
        self.dt = sim.timestep
        self.num_jumps = sim.num_phases
        self.num_frames = sim.total_frames
        self.auction_type = sim.auction_type
        self.use_config = sim.use_config
        self.config_data = sim.config_data
        self.config_order = sim.config_order

        self.steps_per_jump = int(self.num_frames / self.num_jumps)
        self.trajectory = np.zeros((len(agents), self.nframes, 3))
        self.energy = 0.0
        self.delta_v_list = np.array([], dtype=float)
        self.neighbor_history = np.zeros(
            (self.num_jumps, len(agents), len(agents)), dtype=float
        )
        self.config_history = np.zeros(
            (self.num_jumps, len(agents), len(agents)), dtype=float
        )

    def set_simulation_params(self, sim: SimulationParams):
        self.dt = sim.timestep
        self.num_jumps = sim.num_phases
        self.num_frames = sim.total_frames
        self.auction_type = sim.auction_type
        self.config_enabled = sim.use_config
        self.config = sim.config_data
        self.fConfig = sim.config_order

        self.steps_per_jump = int(self.num_frames / self.num_jumps)
        self.trajectory = np.zeros((len(self.agents), self.num_frames, 3))
        self.energy = 0.0
        self.deltavs = []
        self.NeighborsDistanceOT = []
        self.NeighborsConfigOT = []

    def run(self):
        for agent in self.agents:
            agent.update_direction()

        for jump in range(self.num_jumps):
            t0 = self.dt * self.steps_per_jump * jump
            t1 = t0 + self.dt * self.steps_per_jump
            angular_velocity = self.angular_velocity
            phi = compute_state_transition_matrix(
                times=(t0, t1),
                eccentricity=self.eccentricity,
                orbital_radius=self.orbital_radius,
                angular_velocity=angular_velocity,
            )[:3]
            # phi_r = np.array([row[:3] for row in phi])
            # phi_v = np.array([row[3:6] for row in phi])
            phi_r = phi[:3, :3]
            phi_v = phi[:3, 3:6]

            if self.auction_type:
                NeighborsDistance = neighborsDistance(
                    self.neighbors_0,
                    self.neighbor_radius,
                    np.array([a.position for a in self.agents]),
                )
                if self.auction_type not in auction_map:
                    raise ValueError(f"Unknown auction type: {self.auction_type}")
                self.R_f = auction_map[self.auction_type](
                    np.array([a.position for a in self.agents]),
                    self.R_f,
                    NeighborsDistance,
                ).assign()
                for i, agent in enumerate(self.agents):
                    agent.target = self.R_f[i]
                    agent.update_direction()

            if self.config_enabled:
                NeighborsConfig = neighborsConfig(self.config, self.fConfig)

            for i, agent in enumerate(self.agents):
                deltav, _ = agent.step(jump, self.num_jumps, phi_r, phi_v)
                self.energy += np.linalg.norm(deltav)
                self.deltavs.append(np.linalg.norm(deltav))

                for step in range(self.steps_per_jump):
                    r_t, v_t = compute_maneuver_points(
                        r_0=agent.position,
                        v_0=agent.velocity,
                        times=[t0 + self.dt * step, t0],
                        eccentricity=self.eccentricity,
                        orbital_radius=self.orbital_radius,
                        angular_velocity=self.angular_velocity,
                    )
                    idx = jump * self.steps_per_jump + step
                    self.trajectory[agent.id][idx] = r_t
                agent.position = r_t
                agent.velocity = v_t

            if self.auction_type:
                self.NeighborsDistanceOT.append((t0, NeighborsDistance))
            if self.config_enabled:
                self.NeighborsConfigOT.append((t0, NeighborsConfig))

        return (
            self.trajectory,
            np.array(self.deltavs),
            self.energy,
            [
                self.NeighborsDistanceOT if self.auction_type else [],
                self.NeighborsConfigOT if self.config_enabled else [],
            ],
        )
