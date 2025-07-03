# /***********************************************************
# * Copyright (c) 2025                                      *
# * Indian Institute of Technology, Bombay                  *
# * Author(s): Aaron John Sabu, Dwaipayan Mukherjee         *
# * Contact  : aaronjs@g.ucla.edu, dm@ee.iitb.ac.in         *
# ***********************************************************/

import copy
import yaml
import numpy as np
import matplotlib.pyplot as mplplt
import plotly.graph_objects as go

from macro.ifmea import IFMEAEngine
from macro.mirror import (
    generate_target_positions,
    calc_number_of_layers,
    compute_desired_attitudes,
)
from macro.utils import (
    initialize_poses,
    consensus_step,
    to_layer_index,
    to_flat_index,
    NullLogger,
)
from macro.agent import Agent, OrbitalParams, SimulationParams
from macro.glideslope import GlideslopeSimulator


class Macro:
    def __init__(self, config_path="config/default.yaml", *args, **kwargs):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.logger = kwargs.get("logger", NullLogger())
        self._init_params()
        self.ifmea_engine = IFMEAEngine()

    def _init_params(self):
        np.random.seed(self.config["simulation"].get("seed", 0))

        self.num_agents = self.config["formation"]["num_agents"]
        self.num_jumps = self.config["formation"]["num_jumps"]
        self.num_frames = self.config["simulation"]["num_frames"]
        self.timestep = self.config["simulation"]["timestep"]

        self.radius = self.config["orbit"]["radius"]
        self.gravpar = self.config["orbit"]["gravpar"]
        self.eccentricity = self.config["orbit"]["eccentricity"]
        self.ang_vel = np.sqrt(self.gravpar / self.radius**3)
        self.ang_mom = self.radius**2 * self.ang_vel

        self.clear_aperture = self.config["formation"]["clear_aperture"]
        self.parab_radius = self.config["formation"]["parab_radius"]
        self.expansion = self.config["formation"]["expansion"]

        self.neighbors = np.zeros((self.num_agents, self.num_agents), dtype=int)
        self.gains = np.full(
            (self.num_agents + 1, self.num_agents + 1, 3),
            [0.5 * 4e-6, 0.5 * 8e-6, 0.5 * 8e-6],
        )
        self.num_layers, self.layer_lengths = calc_number_of_layers(self.num_agents)

        self.initial_velocities = np.zeros((self.num_agents, 3))
        self.initial_positions, self.initial_attitudes = initialize_poses(self.num_agents, 5.00)
        self.formation_positions, _ = initialize_poses(self.num_agents, 0.01)
        self.unordered_target_positions = generate_target_positions(
            self.clear_aperture,
            self.parab_radius,
            self.num_agents,
            self.layer_lengths,
            self.expansion,
        )
        self.target_positions = self.unordered_target_positions.copy()

        self.stage_changes = {
            "1_rendezvous": None,
            "2_formation": None,
            "3_ifmea": None,
        }
        self.trajectory = np.empty((self.num_agents, 0, 6))
        self.attitudes = np.empty((self.num_agents, 0, 3))
        self.all_neighbors, self.all_flat_config = [], []

        self.true_mapping = [
            list(np.full(l, i + 1)) for i, l in enumerate(self.layer_lengths)
        ]
        self.flat_config = np.arange(self.num_agents)
        self.pre_ifmea_config = [[] for _ in self.layer_lengths]

    def _plot_trajectory(self, title, stage_name):
        fig = go.Figure()
        stage_keys = list(self.stage_changes.keys())
        stage_index = stage_keys.index(stage_name)

        # Get start index for this stage
        start_idx = self.stage_changes[stage_name]

        # Get end index: next stage start or end of trajectory
        if stage_index + 1 < len(stage_keys):
            next_stage = stage_keys[stage_index + 1]
            end_idx = self.stage_changes[next_stage]
        else:
            end_idx = self.trajectory.shape[1]

        for agent in range(self.num_agents):
            fig.add_trace(
                go.Scatter3d(
                    x=self.trajectory[agent, start_idx:end_idx, 0],
                    y=self.trajectory[agent, start_idx:end_idx, 1],
                    z=self.trajectory[agent, start_idx:end_idx, 2],
                    mode="markers",
                    marker=dict(size=3, opacity=0.7),
                    name=f"S/c {agent + 1}",
                )
            )
        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",
            ),
            legend_title_text="Spacecraft",
            margin=dict(l=0, r=0, b=0, t=30),
            title_text=f"3D Trajectories: {stage_name.replace('_', ' ').title()}",
            template="plotly_dark",
        )
        fig.write_html(title, auto_open=False)

    def _plot_attitude(self):
        mplplt.style.use("fivethirtyeight")
        labels = ["Roll", "Pitch", "Yaw"]
        fig, axes = mplplt.subplots(
            3, 1, figsize=(12, 10), sharex=True, constrained_layout=True
        )
        for dim, ax in enumerate(axes):
            for agent in range(self.num_agents):
                ax.plot(
                    self.attitudes[agent, :, dim],
                    label=f"Agent {agent + 1}",
                )
            ax.set_ylabel(labels[dim])
            ax.legend(loc="upper right", fontsize="small")
            ax.grid(True, linestyle="--", alpha=0.6)
        axes[-1].set_xlabel("Time Step")
        fig.suptitle("Attitude over Time")
        fig.savefig("./results/attitude_consensus.png", dpi=300)
        mplplt.close(fig)

    def _simulate_phase(self, start, end, stage_name, auction_type, plot=True):
        agents = [
            Agent(
                agent_id=agent,
                agent_type=1,
                position=start[agent],
                velocity=self.initial_velocities[agent],
                target=end[agent],
            )
            for agent in range(self.num_agents)
        ]
        orbit_params = OrbitalParams(
            radius=self.radius,
            eccentricity=self.eccentricity,
            angular_momentum=self.ang_mom,
            angular_velocity=self.ang_vel,
        )
        sim_params = SimulationParams(
            num_jumps=self.num_jumps,
            timestep=self.timestep,
            num_frames=self.num_frames,
            auction_type=auction_type,
            layer_config=self.true_mapping,
            flat_config=self.flat_config,
        )
        sim = GlideslopeSimulator(
            agents=agents,
            neighbor_matrix=self.neighbors,
            neighbor_radius=self.config["formation"]["neighbor_radius"],
            target_positions=end,
            orbit=orbit_params,
            sim=sim_params,
        )
        traj, _, _, _ = sim.run()
        traj = np.array(traj)

        self.trajectory = np.concatenate((self.trajectory, traj), axis=1)
        
        if plot:
            self._plot_trajectory(f"./results/{stage_name}.html", stage_name)
        return sim.get_last_system_graph()

    def _generate_ifmea_config(self):
        for agent in range(self.num_agents):
            layer_idx, _ = to_layer_index(
                self.layer_lengths,
                np.argmin(
                    np.linalg.norm(
                        self.unordered_target_positions
                        - self.trajectory[agent, -1, :3],
                        axis=1,
                    )
                ),
            )
            self.pre_ifmea_config[layer_idx].append(self.flat_config[agent])

    def _intra_formation_exchange(self):
        def apply_rotation(layer_idx, shift):
            start_idx = sum(self.layer_lengths[:layer_idx])
            end_idx = start_idx + self.layer_lengths[layer_idx]
            block = self.flat_config[start_idx:end_idx]
            self.flat_config[start_idx:end_idx] = np.concatenate(
                (block[-shift:], block[:-shift])
            )

        def apply_exchange(index_triplet):
            agents = [to_flat_index(self.layer_lengths, idx[0], idx[1]) for idx in index_triplet]
            (
                self.flat_config[agents[0]],
                self.flat_config[agents[1]],
                self.flat_config[agents[2]],
            ) = (agents[2], agents[0], agents[1])

        intermediate_ifmea_positions = self.formation_positions.copy()
        target_positions = self.formation_positions.copy()

        self.ifmea_engine.set_layer_config(self.pre_ifmea_config)
        commands = self.ifmea_engine.run()

        for command in commands:
            cmd_type = command[0]

            if cmd_type == "R":
                layer_idx, shift = command[1], command[2]
                apply_rotation(layer_idx, shift)
            elif cmd_type == "E":
                index_triplet = command[1:4]
                apply_exchange(index_triplet)
            self.all_flat_config.append(self.flat_config.copy())

            intermediate_ifmea_positions = target_positions.copy()
            for idx, agent in enumerate(self.flat_config):
                target_positions[agent] = self.unordered_target_positions[idx]

            system_graph = self._simulate_phase(
                start=intermediate_ifmea_positions,
                end=target_positions,
                stage_name="3_ifmea",
                auction_type=None,
                plot=False,
            )
            self.all_neighbors.append(system_graph.get_all_neighbors())

        self.target_positions = target_positions.copy()
        self._plot_trajectory("./results/3_ifmea.html", "3_ifmea")
        return self.all_neighbors, self.all_flat_config

    def _attitude_consensus(self):
        num_steps = self.trajectory.shape[1] - self.stage_changes["3_ifmea"]
        self.attitudes = np.zeros((self.num_agents, num_steps, 3), dtype=float)

        # Initialize all agents at t=0 with their initial attitudes
        self.attitudes[:, 0, :] = self.initial_attitudes

        # Compute the target/desired attitude
        desired_attitudes = compute_desired_attitudes(
            self.clear_aperture,
            self.parab_radius,
            self.layer_lengths,
            self.flat_config,
        )

        # Run consensus over each time step
        for frame in range(num_steps - 1):
            for agent_id in range(self.num_agents):
                self.attitudes[agent_id, frame + 1, :] = consensus_step(
                    agent_id=agent_id,
                    neighbors=self.all_neighbors[-1].get(agent_id, []),
                    previous=self.attitudes[:, frame, :],
                    desired=desired_attitudes,
                    gains=self.gains[agent_id],
                    timestep=self.timestep,
                    count=frame,
                )
        
        self._plot_attitude()

    def run(self):
        self.stage_changes["1_rendezvous"] = 0
        self._simulate_phase(
            start=self.initial_positions,
            end=self.formation_positions,
            stage_name="1_rendezvous",
            auction_type="Hybrid",
        )
        self.logger.info(
            f"Rendezvous completed. Trajectory shape: {self.trajectory.shape}"
        )

        self.stage_changes["2_formation"] = self.trajectory.shape[1] - 1
        self._simulate_phase(
            start=self.formation_positions,
            end=self.unordered_target_positions,
            stage_name="2_formation",
            auction_type="Hybrid",
        )
        self.logger.info(
            f"Formation completed. Trajectory shape: {self.trajectory.shape}"
        )
        
        self.stage_changes["3_ifmea"] = self.trajectory.shape[1] - 1
        self._generate_ifmea_config()
        self._intra_formation_exchange()
        self.logger.info(
            f"IFMEA completed. Trajectory shape: {self.trajectory.shape}"
        )

        self._attitude_consensus()
        self.logger.info("Attitude consensus completed.")


if __name__ == "__main__":
    raise NotImplementedError(
        "This script is not intended to be run directly. Use the run.py script instead."
    )
