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

import copy
import numpy as np

from macro.simulate import simulatePL
from macro.glideslope import GlideslopeSimulation
from macro.agent import Agent, OrbitalParams, SimulationParams
from macro.utils import init_pose, load_config


class BenchmarkRunner:
    def __init__(self, num_agents=18, num_jumps=5):
        self.num_agents = num_agents
        self.num_jumps = num_jumps
        self.config = load_config()
        self.orbit = self._compute_orbital_params()
        self.sim_params = self._create_simulation_params()
        self.R_0 = self._get_initial_positions()
        self.R_f = self._generate_target_positions()
        self.neighbors = self._init_neighbor_matrix()

    def _compute_orbital_params(self):
        return OrbitalParams(
            radius=self.config["orbit"]["radius"],
            gravpar=self.config["orbit"]["gravpar"],
            eccentricity=self.config["orbit"]["eccentricity"],
            ang_vel=np.sqrt(
                self.config["orbit"]["gravpar"] / self.config["orbit"]["radius"] ** 3
            ),
            ang_mom=(self.config["orbit"]["radius"] ** 2)
            * np.sqrt(
                self.config["orbit"]["gravpar"] / self.config["orbit"]["radius"] ** 3
            ),
        )

    def _create_simulation_params(self):
        return SimulationParams(
            num_jumps=self.num_jumps,
            timestep=self.config["simulation"]["timestep"],
            num_frames=self.config["simulation"]["num_frames"],
            auction_type=self.config["simulation"]["auction_type"],
            use_config=self.config["simulation"]["use_config"],
            layer_config=[[1, 2, 1, 1, 1, 2], [2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2]],
            flat_config=[i for i in range(self.num_agents)],
        )

    def _get_initial_positions(self):
        return np.array(
            [
                [2.29364301, 1.26911104, -1.71273596],
                [-2.30499439, 1.46048986, 0.84474138],
                [-2.20553178, 0.33463571, -0.45545043],
                [-2.00737466, 2.69883730, 2.20860278],
                [-2.13433824, -2.45255776, 0.01869546],
                [-2.87274879, -1.20121056, -2.55504595],
                [-2.82212589, -2.88499068, -2.82692798],
                [-2.32413586, 1.38993451, -1.85659537],
                [-2.79356145, -1.84074991, 1.31134734],
                [-2.45089991, 2.86245750, -0.92419686],
                [-2.15760877, -1.63576172, -0.10413470],
                [-2.22633934, 2.55351914, -2.75085611],
                [-2.19934209, -1.28014131, 1.85593947],
                [-2.72048638, 1.75400878, -1.49284592],
                [-2.66459536, 0.56970289, -1.64283741],
                [-2.26072675, 1.64575550, -1.69458091],
                [-2.34255969, 2.06506644, -1.14374557],
                [-2.99163571, -0.24707069, 1.72698837],
            ]
        )

    def _generate_target_positions(self, inner_radius=0.4):
        return np.array(
            [
                [
                    inner_radius * 1 * np.sin(np.deg2rad(0)),
                    inner_radius * 1 * np.cos(np.deg2rad(0)),
                    0.000,
                ],
                [
                    inner_radius * 1 * np.sin(np.deg2rad(60)),
                    inner_radius * 1 * np.cos(np.deg2rad(60)),
                    0.000,
                ],
                [
                    inner_radius * 1 * np.sin(np.deg2rad(120)),
                    inner_radius * 1 * np.cos(np.deg2rad(120)),
                    0.000,
                ],
                [
                    inner_radius * 1 * np.sin(np.deg2rad(180)),
                    inner_radius * 1 * np.cos(np.deg2rad(180)),
                    0.000,
                ],
                [
                    inner_radius * 1 * np.sin(np.deg2rad(240)),
                    inner_radius * 1 * np.cos(np.deg2rad(240)),
                    0.000,
                ],
                [
                    inner_radius * 1 * np.sin(np.deg2rad(300)),
                    inner_radius * 1 * np.cos(np.deg2rad(300)),
                    0.000,
                ],
                [
                    inner_radius * 2 * np.sin(np.deg2rad(0)),
                    inner_radius * 2 * np.cos(np.deg2rad(0)),
                    0.000,
                ],
                [
                    inner_radius * 2 * np.sin(np.deg2rad(30)),
                    inner_radius * 2 * np.cos(np.deg2rad(30)),
                    0.000,
                ],
                [
                    inner_radius * 2 * np.sin(np.deg2rad(60)),
                    inner_radius * 2 * np.cos(np.deg2rad(60)),
                    0.000,
                ],
                [
                    inner_radius * 2 * np.sin(np.deg2rad(90)),
                    inner_radius * 2 * np.cos(np.deg2rad(90)),
                    0.000,
                ],
                [
                    inner_radius * 2 * np.sin(np.deg2rad(120)),
                    inner_radius * 2 * np.cos(np.deg2rad(120)),
                    0.000,
                ],
                [
                    inner_radius * 2 * np.sin(np.deg2rad(150)),
                    inner_radius * 2 * np.cos(np.deg2rad(150)),
                    0.000,
                ],
                [
                    inner_radius * 2 * np.sin(np.deg2rad(180)),
                    inner_radius * 2 * np.cos(np.deg2rad(180)),
                    0.000,
                ],
                [
                    inner_radius * 2 * np.sin(np.deg2rad(210)),
                    inner_radius * 2 * np.cos(np.deg2rad(210)),
                    0.000,
                ],
                [
                    inner_radius * 2 * np.sin(np.deg2rad(240)),
                    inner_radius * 2 * np.cos(np.deg2rad(240)),
                    0.000,
                ],
                [
                    inner_radius * 2 * np.sin(np.deg2rad(270)),
                    inner_radius * 2 * np.cos(np.deg2rad(270)),
                    0.000,
                ],
                [
                    inner_radius * 2 * np.sin(np.deg2rad(300)),
                    inner_radius * 2 * np.cos(np.deg2rad(300)),
                    0.000,
                ],
                [
                    inner_radius * 2 * np.sin(np.deg2rad(330)),
                    inner_radius * 2 * np.cos(np.deg2rad(330)),
                    0.000,
                ],
            ]
        )

    def _init_neighbor_matrix(self):
        neighbors = [
            [0 for _ in range(self.num_agents)] for _ in range(self.num_agents)
        ]
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j:
                    dist = np.linalg.norm(self.R_0[i] - self.R_0[j])
                    if dist < self.sim_params.neighbor_radius:
                        neighbors[i][j] = 1
        return neighbors

    def run(self, auction_types):
        simulation = GlideslopeSimulation(
            agents=[
                Agent(
                    agent_id=i,
                    agent_type=1,
                    position=self.R_0[i],
                    velocity=np.zeros(3),
                    target=self.R_f[i],
                )
                for i in range(self.num_agents)
            ],
            neighbors_matrix=self.neighbors,
            neighbor_radius=self.config["simulation"]["neighbor_radius"],
            target_positions=self.R_f,
            orbit=self.orbit,
            sim=self.sim_params,
        )
        for auction_type in auction_types:
            self.sim_params.auction_type = auction_type
            simulation.set_simulation_params(self.sim_params)
            traj, deltav, energy, neighbor_data = simulation.run()
            simulatePL(
                f"benchmark-{auction_type}.html",
                traj[:, :, 0],
                traj[:, :, 1],
                traj[:, :, 2],
                [f"Spacecraft {i}" for i in range(self.num_agents)],
            )
