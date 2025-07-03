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
from macro.utils import init_pose, consensus_step, to_layer_index, NullLogger
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
        self.agent_params = np.full(
            (self.num_agents + 1, self.num_agents + 1, 3),
            [0.5 * 4e-6, 0.5 * 8e-6, 0.5 * 8e-6],
        )
        self.num_layers, self.layer_lens = calc_number_of_layers(self.num_agents)

        self.R_f0 = generate_target_positions(
            self.clear_aperture,
            self.parab_radius,
            self.num_agents,
            self.layer_lens,
            self.expansion,
        )
        self.R_f = self.R_f0.copy()
        self.V_0 = np.zeros((self.num_agents, 3))
        self.R_0, self.Theta = init_pose(self.num_agents, 5.00)
        self.R_I, _ = init_pose(self.num_agents, 0.01)

        self.true_mapping = [np.full(l, i + 1) for i, l in enumerate(self.layer_lens)]
        self.flat_config = np.arange(self.num_agents)

    def _simulate_phase(self, start, end, stage_name, auction_type):
        agents = [
            Agent(
                agent_id=i,
                agent_type=1,
                position=start[i],
                velocity=self.V_0[i],
                target=end[i],
            )
            for i in range(self.num_agents)
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
        self._plot(f"{stage_name}.html", traj[:, :, 0], traj[:, :, 1], traj[:, :, 2])
        return (
            traj[:, :, 0],
            traj[:, :, 1],
            traj[:, :, 2],
            sim.get_last_system_graph(),
        )

    def _plot(self, title, X, Y, Z):
        fig = go.Figure()
        for i, (x, y, z) in enumerate(zip(X, Y, Z)):
            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=dict(size=4, opacity=0.8),
                    name=f"S/c {i}",
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
            title_text="3D Spacecraft Trajectories",
            template="plotly_dark",
        )
        fig.write_html(title, auto_open=False)

    def _plot_attitude_angles(self):
        mplplt.style.use("fivethirtyeight")
        labels = ["Roll", "Pitch", "Yaw"]
        fig, axes = mplplt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for dim, ax in enumerate(axes):
            for agent in range(self.num_agents):
                ax.plot(
                    [theta[dim] for theta in self.Theta[agent]],
                    label=f"Agent {agent + 1}",
                )
            ax.set_ylabel(labels[dim])
            ax.legend(loc="upper right", fontsize="small")
            ax.grid(True, linestyle="--", alpha=0.6)
        axes[-1].set_xlabel("Time Step")
        fig.suptitle("Roll, Pitch, and Yaw Angles for Each Agent")
        mplplt.tight_layout(rect=[0, 0, 1, 0.97])
        mplplt.show()

    def _intra_formation_exchange(self, current_config):
        traj = np.empty((self.num_agents, 3), dtype=object)
        for dim in range(3):
            for agent in range(self.num_agents):
                traj[agent, dim] = []
        all_neighbors, all_flat_config = [], []
        R_0 = self.R_f
        self.ifmea_engine.set_layer_config(self.layer_config)
        commands = self.ifmea_engine.run()

        for command in commands:
            for _ in range(command[4]):
                if command[0] == "R":
                    r = self.flat_config[command[1] : command[2] + 1]
                    r = np.concatenate((r[-command[3] :], r[: -command[3]]))
                    self.flat_config = np.concatenate(
                        (
                            self.flat_config[: command[1]],
                            r,
                            self.flat_config[command[2] + 1 :],
                        )
                    )
                elif command[0] == "E":
                    R_0 = self.R_f.copy()
                    i1, i2, i3 = [self.flat_config[i] for i in command[1:4]]
                    self.R_f[[i1, i2, i3]] = self.R_f[[i2, i3, i1]]
                    (
                        self.flat_config[command[1]],
                        self.flat_config[command[2]],
                        self.flat_config[command[3]],
                    ) = (i3, i1, i2)

                all_flat_config.append(self.flat_config.copy())
                for idx, agent in enumerate(self.flat_config):
                    self.R_f[agent] = self.R_f0[idx]

                X, Y, Z, system_graph = self._simulate_phase(
                    start=R_0,
                    end=self.R_f,
                    stage_name="Stage-02",
                    auction_type=None,
                )
                all_neighbors.append(system_graph.get_all_neighbors())
                R_0 = self.R_f
                for i in range(self.num_agents):
                    traj[i, 0].extend(X[i])
                    traj[i, 1].extend(Y[i])
                    traj[i, 2].extend(Z[i])

        self._plot("Stage-02.html", traj[:, 0], traj[:, 1], traj[:, 2])
        return all_neighbors, all_flat_config

    def _attitude_consensus(self, all_neighbors, all_flat_config):
        for k in range(len(all_neighbors)):
            theta_desired = compute_desired_attitudes(
                self.clear_aperture,
                self.parab_radius,
                self.layer_lens,
                all_flat_config[k],
            )
            neighbors = all_neighbors[k]
            for i in range(self.num_frames):
                Theta_copy = self.Theta.copy()
                for agent_id in range(self.num_agents):
                    self.Theta[agent_id].append(
                        consensus_step(
                            agent_index=agent_id,
                            value_history=Theta_copy,
                            desired_value=theta_desired,
                            neighbor_indices=neighbors[agent_id],
                            control_gains=self.agent_params[agent_id],
                            timestep=self.timestep,
                            iteration_count=i,
                        )
                    )
        self._plot_attitude_angles()

    def run(self):
        _, _, _, _ = self._simulate_phase(
            start=self.R_0, end=self.R_I, stage_name="Stage-01A", auction_type="Hybrid"
        )
        X, Y, Z, _ = self._simulate_phase(
            start=self.R_I,
            end=self.R_f,
            stage_name="Stage-01B",
            auction_type="Hybrid",
        )

        present = [[] for _ in range(len(self.layer_lens))]
        for agent in range(self.num_agents):
            r_0 = np.array([X[agent, -1], Y[agent, -1], Z[agent, -1]])
            idx, _ = to_layer_index(
                self.layer_lens, np.argmin(np.linalg.norm(self.R_f - r_0, axis=1))
            )
            present[idx].append(self.true_mapping[agent])

        all_neighbors, all_flat_config = self._intra_formation_exchange(present)
        self._attitude_consensus(all_neighbors, all_flat_config)


if __name__ == "__main__":
    raise NotImplementedError(
        "This script is not intended to be run directly. Use the run.py script instead."
    )
