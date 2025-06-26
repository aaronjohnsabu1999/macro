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
from macro.maneuvers import getTransformMatrixTH, getManeuverPointsTH
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

    def step(self, index, total_steps, phi_r, phi_v):
        if np.linalg.norm(self.glideslope_dir) < 1e-4:
            return np.zeros(3), []

        scale = 1.0 - (index + 1) / total_steps
        r_k1 = (
            self.target
            + scale * np.linalg.norm(self.position - self.target) * self.glideslope_dir
        )
        deltav = np.linalg.inv(phi_v) @ (r_k1 - phi_r @ self.position) - self.velocity
        self.velocity += deltav
        return deltav, []  # Maneuver points added externally


class GlideslopeSimulator:
    def __init__(
        self,
        agents,
        neighbors_0,
        neighbor_radius,
        R_T,
        R,
        e,
        h,
        omega,
        num_jumps,
        dt,
        nframes,
        auction_type="Distributed",
        config_enabled=False,
        config=(),
        fConfig=(),
    ):
        self.agents = agents
        self.neighbors_0 = neighbors_0
        self.neighbor_radius = neighbor_radius
        self.R_T = R_T
        self.R = R
        self.e = e
        self.h = h
        self.omega = omega
        self.num_jumps = num_jumps
        self.dt = dt
        self.nframes = nframes
        self.auction_type = auction_type
        self.config_enabled = config_enabled
        self.config = config
        self.fConfig = fConfig

        self.steps_per_jump = int(nframes / num_jumps)
        self.trajectory = {agent.id: np.zeros((nframes, 3)) for agent in agents}
        self.energy = 0.0
        self.deltavs = []
        self.NeighborsDistanceOT = []
        self.NeighborsConfigOT = []

    def run(self):
        for agent in self.agents:
            agent.update_direction()

        for jump in range(self.num_jumps):
            t = self.dt * self.steps_per_jump * jump
            tnext = t + self.dt * self.steps_per_jump
            f, fnext = time_to_true_anomaly(self.omega, self.e, [t, tnext])
            phi = getTransformMatrixTH(self.R, self.e, self.h, [tnext, t], [fnext, f])[
                :3
            ]
            phi_r = np.array([row[:3] for row in phi])
            phi_v = np.array([row[3:6] for row in phi])

            if self.auction_type:
                NeighborsDistance = neighborsDistance(
                    self.neighbors_0,
                    self.neighbor_radius,
                    np.array([a.position for a in self.agents]),
                )
                if self.auction_type not in auction_map:
                    raise ValueError(f"Unknown auction type: {self.auction_type}")
                self.R_T = auction_map[self.auction_type](
                    np.array([a.position for a in self.agents]),
                    self.R_T,
                    NeighborsDistance,
                ).assign()
                for i, agent in enumerate(self.agents):
                    agent.target = self.R_T[i]
                    agent.update_direction()

            if self.config_enabled:
                NeighborsConfig = neighborsConfig(self.config, self.fConfig)

            for i, agent in enumerate(self.agents):
                deltav, _ = agent.step(jump, self.num_jumps, phi_r, phi_v)
                self.energy += np.linalg.norm(deltav)
                self.deltavs.append(np.linalg.norm(deltav))

                for step in range(self.steps_per_jump):
                    time = self.dt * step
                    rt, vt = getManeuverPointsTH(
                        agent.position,
                        agent.velocity,
                        [t + time, t],
                        self.R,
                        self.e,
                        self.h,
                    )
                    idx = jump * self.steps_per_jump + step
                    self.trajectory[agent.id][idx] = rt
                agent.position = rt
                agent.velocity = vt

            if self.auction_type:
                self.NeighborsDistanceOT.append((t, NeighborsDistance))
            if self.config_enabled:
                self.NeighborsConfigOT.append((t, NeighborsConfig))

        return (
            self.trajectory,
            np.array(self.deltavs),
            self.energy,
            [
                self.NeighborsDistanceOT if self.auction_type else [],
                self.NeighborsConfigOT if self.config_enabled else [],
            ],
        )
