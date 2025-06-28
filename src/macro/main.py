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

from macro.ifmea import ifmea, listPointToTwoDimPoint
from macro.mirror import genR_f, layerCalc, calcTheta_d
from macro.glideslope import Agent, GlideslopeSimulator
from macro.utils import init_pose, consensus_step, NullLogger


class Macro:
    def __init__(self, config_path="config/default.yaml", *args, **kwargs):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.logger = kwargs.get("logger", NullLogger())
        self._init_params()

    def _init_params(self):
        sim = self.config["simulation"]
        form = self.config["formation"]
        orbit = self.config["orbit"]

        np.random.seed(sim.get("seed", 0))

        self.num_agents = form["num_agents"]
        self.num_jumps = form["num_jumps"]
        self.num_frames = sim["num_frames"]
        self.dt = sim["dt"]

        self.radius = orbit["radius"]
        self.gravpar = orbit["gravpar"]
        self.eccentricity = orbit["eccentricity"]
        self.ang_vel = np.sqrt(self.gravpar / self.radius**3)
        self.ang_mom = self.radius**2 * self.ang_vel

        self.clear_aperture = form["clear_aperture"]
        self.parab_radius = form["parabolic_radius"]
        self.expansion = form["expansion"]

        self.neighbors = np.zeros((self.num_agents, self.num_agents), dtype=int)
        self.agent_params = np.full(
            (self.num_agents + 1, self.num_agents + 1, 3),
            [0.5 * 4e-6, 0.5 * 8e-6, 0.5 * 8e-6],
        )
        self.num_layers, self.layer_lens = layerCalc(self.num_agents)

        self.R_f0 = genR_f(
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

        self.individuals = np.arange(self.num_agents)
        self.true_mapping = np.concatenate(
            [np.full(l, i + 1) for i, l in enumerate(self.layer_lens)]
        )

    def _simulate_phase(self, start, end, stage_name, auction):
        agents = [
            Agent(id=i, position=start[i], velocity=self.V_0[i], target=end[i])
            for i in range(self.num_agents)
        ]
        sim = GlideslopeSimulator(
            agents,
            self.neighbors,
            self.config["formation"]["neighbor_radius"],
            start,
            self.V_0,
            end,
            self.radius,
            self.eccentricity,
            self.ang_mom,
            self.ang_vel,
            self.num_jumps,
            self.dt,
            self.num_frames,
            auction_type=auction[0],
            config_enabled=auction[1],
            config=self.true_mapping,
            fConfig=self.individuals,
        )
        traj, *_ = sim.run()
        traj = np.array(list(traj.values()))  # ensure NumPy array conversion
        self._plot(f"{stage_name}.html", traj[:, :, 0], traj[:, :, 1], traj[:, :, 2])
        return (
            traj[:, :, 0],
            traj[:, :, 1],
            traj[:, :, 2],
            sim.NeighborsDistanceOT,
            sim.NeighborsConfigOT,
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

    def _intra_formation_exchange(self, present_config):
        Xs, Ys, Zs = np.empty((3, self.num_agents), dtype=object)
        for dim in range(3):
            for agent in range(self.num_agents):
                Xs[dim, agent] = []
        AllNeighbors, FConfig = [], []
        R_0 = self.R_f
        IFMEA = ifmea(present_config)

        for command in IFMEA:
            for _ in range(command[4]):
                if command[0] == "R":
                    r = self.individuals[command[1] : command[2] + 1]
                    r = np.concatenate((r[-command[3] :], r[: -command[3]]))
                    self.individuals = np.concatenate(
                        (
                            self.individuals[: command[1]],
                            r,
                            self.individuals[command[2] + 1 :],
                        )
                    )
                elif command[0] == "E":
                    R_0 = self.R_f.copy()
                    i1, i2, i3 = [self.individuals[i] for i in command[1:4]]
                    self.R_f[[i1, i2, i3]] = self.R_f[[i2, i3, i1]]
                    (
                        self.individuals[command[1]],
                        self.individuals[command[2]],
                        self.individuals[command[3]],
                    ) = (i3, i1, i2)

                FConfig.append(self.individuals.copy())
                for idx, agent in enumerate(self.individuals):
                    self.R_f[agent] = self.R_f0[idx]

                X, Y, Z, _, out = self._simulate_phase(
                    R_0, self.R_f, "Stage-02", (None, True)
                )
                AllNeighbors.append(out[1])
                R_0 = self.R_f
                for i in range(self.num_agents):
                    Xs[0, i].extend(X[i])
                    Xs[1, i].extend(Y[i])
                    Xs[2, i].extend(Z[i])

        self._plot("Stage-02.html", Xs[0], Xs[1], Xs[2])
        return AllNeighbors, FConfig

    def _attitude_consensus(self, AllNeighbors, FConfig):
        for k in range(len(AllNeighbors)):
            Theta_d = calcTheta_d(
                self.clear_aperture, self.parab_radius, self.layer_lens, FConfig[k]
            )
            for i in range(self.num_frames):
                Neighbors = AllNeighbors[k][0][1]
                tempTheta = [
                    consensus_step(
                        agent,
                        self.Theta,
                        Theta_d,
                        Neighbors[agent],
                        self.agent_params[agent],
                        self.dt,
                        i,
                    )
                    for agent in range(self.num_agents)
                ] + [[0.0, 0.0, 0.0]]
                for agent in range(self.num_agents):
                    self.Theta[agent].append(tempTheta[agent])
        self._plot_attitude_angles()

    def run(self):
        _, _, _, _ = self._simulate_phase(
            self.R_0, self.R_I, "Stage-01A", ("Hybrid", False)
        )
        X, Y, Z, _, _ = self._simulate_phase(
            self.R_I, self.R_f, "Stage-01B", ("Hybrid", False)
        )

        present = [[] for _ in range(len(self.layer_lens))]
        for agent in range(self.num_agents):
            r_0 = np.array([X[agent, -1], Y[agent, -1], Z[agent, -1]])
            idx, _ = listPointToTwoDimPoint(
                self.layer_lens, np.argmin(np.linalg.norm(self.R_f - r_0, axis=1))
            )
            present[idx].append(self.true_mapping[agent])

        AllNeighbors, FConfig = self._intra_formation_exchange(present)
        self._attitude_consensus(AllNeighbors, FConfig)


if __name__ == "__main__":
    raise NotImplementedError(
        "This script is not intended to be run directly. Use the run.py script instead."
    )
