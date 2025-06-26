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
import yaml
import numpy as np
import matplotlib.pyplot as mplplt
import plotly.offline as ptyplt
import plotly.graph_objects as go
from matplotlib import animation

from macro.ifmea import ifmea, listPointToTwoDimPoint
from macro.mirror import genR_T, layerCalc, calcTheta_d
from macro.simulate import plotRPY, simulatePL, simulateMPL
from macro.glideslope import multiAgentGlideslope
from macro.utils import init_pose, consensus_step

mplplt.style.use("fivethirtyeight")


class MacroSimulation:
  def __init__(self, config_path="config/default.yaml"):
    with open(config_path, "r") as f:
      self.config = yaml.safe_load(f)
    self._init_params()

  def _init_params(self):
    sim = self.config["simulation"]
    form = self.config["formation"]
    orbit = self.config["orbit"]

    np.random.seed(sim.get("seed", 0))

    self.num_agents = form["num_agents"]
    self.num_jumps = form["num_jumps"]
    self.nframes = sim["num_frames"]
    self.dt = sim["dt"]

    self.radius = orbit["radius"]
    self.mu = orbit["gravpar"]
    self.e = orbit["eccentricity"]
    self.omega = np.sqrt(self.mu / self.radius**3)
    self.h = self.radius**2 * self.omega

    self.clear_aperture = form["clear_aperture"]
    self.parab_radius = form["parabolic_radius"]
    self.expansion = form["expansion"]

    self.neighbors = [[0] * self.num_agents for _ in range(self.num_agents)]
    self.agent_params = [
        [[0.5 * 4e-6, 0.5 * 8e-6, 0.5 * 8e-6]
         for _ in range(self.num_agents + 1)]
        for _ in range(self.num_agents + 1)
    ]
    self.num_layers, self.layer_lens = layerCalc(self.num_agents)

    self.R_T0 = genR_T(
        self.clear_aperture,
        self.parab_radius,
        self.num_agents,
        self.layer_lens,
        self.expansion,
    )
    self.R_T = self.R_T0.copy()
    self.V_0 = np.zeros((self.num_agents, 3))
    self.R_0, self.Theta = init_pose(self.num_agents, 5.00)
    self.R_I, _ = init_pose(self.num_agents, 0.01)

    self.individuals = list(range(self.num_agents))
    self.true_mapping = [
        layer
        for layer in range(1, self.num_layers + 1)
        for _ in range(self.layer_lens[layer - 1])
    ]

  def _simulate_phase(self, start, end, stage_name, auction):
    X, Y, Z, *_, output = multiAgentGlideslope(
        self.neighbors,
        self.config["formation"]["neighbor_radius"],
        start,
        self.V_0,
        end,
        self.radius,
        self.e,
        self.h,
        self.omega,
        self.num_jumps,
        self.dt,
        self.nframes,
        auctionParams=auction,
        Config=(self.true_mapping, self.individuals),
    )
    self._simulatePL(f"{stage_name}.html", X, Y, Z)
    return X, Y, Z, output

  def _simulatePL(self, title, X, Y, Z):
    fig = go.Figure(
        data=[
            go.Scatter3d(
                mode="markers",
                x=X[i],
                y=Y[i],
                z=Z[i],
                marker=dict(size=3),
                name=f"S/c {i}",
            )
            for i in range(len(X))
        ]
    )
    ptyplt.plot(fig, filename=title)

  def _plotRPY(self):
    for dim in range(3):
      for agent in range(self.num_agents):
        mplplt.plot(
            [self.Theta[agent][i][dim] for i in range(len(self.Theta[agent]))]
        )
      mplplt.show()

  def _intra_formation_exchange(self, present_config):
    Xs, Ys, Zs = [[[] for _ in range(self.num_agents)] for _ in range(3)]
    AllNeighbors, FConfig = [], []
    R_0 = self.R_T
    IFMEA = ifmea(present_config)

    for command in IFMEA:
      for _ in range(command[4]):
        if command[0] == "R":
          r = self.individuals[command[1]: command[2] + 1]
          r = r[-command[3]:] + r[: -command[3]]
          self.individuals = (
              self.individuals[: command[1]]
              + r
              + self.individuals[command[2] + 1:]
          )
        elif command[0] == "E":
          R_0 = self.R_T.copy()
          i1, i2, i3 = [self.individuals[i] for i in command[1:4]]
          self.R_T[i1], self.R_T[i2], self.R_T[i3] = (
              self.R_T[i2],
              self.R_T[i3],
              self.R_T[i1],
          )
          (
              self.individuals[command[1]],
              self.individuals[command[2]],
              self.individuals[command[3]],
          ) = (i3, i1, i2)

        FConfig.append(self.individuals.copy())
        for agent in self.individuals:
          self.R_T[agent] = self.R_T0[self.individuals.index(agent)]

        X, Y, Z, *_, out = multiAgentGlideslope(
            self.neighbors,
            self.config["formation"]["neighbor_radius"],
            R_0,
            self.V_0,
            self.R_T,
            self.radius,
            self.e,
            self.h,
            self.omega,
            self.num_jumps,
            self.dt,
            self.nframes,
            auctionParams=(False, "", True),
            Config=(present_config, self.individuals),
        )
        AllNeighbors.append(out[1])
        R_0 = self.R_T
        for i in range(self.num_agents):
          Xs[i].extend(X[i])
          Ys[i].extend(Y[i])
          Zs[i].extend(Z[i])

    self._simulatePL("Stage-02.html", Xs, Ys, Zs)
    return AllNeighbors, FConfig

  def _attitude_consensus(self, AllNeighbors, FConfig):
    for k in range(len(AllNeighbors)):
      Theta_d = calcTheta_d(
          self.clear_aperture, self.parab_radius, self.layer_lens, FConfig[k]
      )
      for i in range(self.nframes):
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
    self._plotRPY()

  def run(self):
    # Phase 1: Initial to Intermediate
    _, _, _, _ = self._simulate_phase(
        self.R_0, self.R_I, "Stage-01A", (True, "Hybrid", False)
    )

    # Phase 2: Intermediate to Target
    X, Y, Z, _ = self._simulate_phase(
        self.R_I, self.R_T, "Stage-01B", (True, "Hybrid", False)
    )

    # Configuration Analysis
    present = [[] for _ in range(len(self.layer_lens))]
    for agent in range(self.num_agents):
      r_0 = np.array([X[agent][-1], Y[agent][-1], Z[agent][-1]])
      r_t = np.array(self.R_T)
      idx, _ = listPointToTwoDimPoint(
          self.layer_lens, np.argmin([np.linalg.norm(r_0 - r) for r in r_t])
      )
      present[idx].append(self.true_mapping[agent])

    # Mutual Exchanges and Consensus
    AllNeighbors, FConfig = self._intra_formation_exchange(present)
    self._attitude_consensus(AllNeighbors, FConfig)


if __name__ == "__main__":
  MacroSimulation().run()
