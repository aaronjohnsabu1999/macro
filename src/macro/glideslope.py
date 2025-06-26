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

from macro.auction import greedy, cnsAuction, disAuction, hybAuction, stdAuction
from macro.utils import time_to_true_anomaly
from macro.maneuvers import getTransformMatrixTH, getManeuverPointsTH
from macro.neighbors import neighborsConfig, neighborsDistance


def reduction_factor(index: int, total_steps: int) -> float:
  """
  Compute a linear reduction factor based on step index.

  Parameters:
      index (int): Current step or iteration index (0-based).
      total_steps (int): Total number of steps or iterations.

  Returns:
      float: Reduction factor in [0, 1), decreasing with index.
  """
  return 1.0 - (index + 1) / total_steps


def scaled_distance_reduction(
    index: int,
    total_steps: int,
    target_position: np.ndarray,
    current_position: np.ndarray,
    initial_position: np.ndarray = None,
) -> float:
  """
  Compute the reduction-scaled distance from current to target position.

  Parameters:
      index (int): Current iteration index.
      total_steps (int): Total number of iterations.
      target_position (np.ndarray): Target 3D position vector.
      current_position (np.ndarray): Current 3D position vector.
      initial_position (np.ndarray, optional): Not used here but accepted for API compatibility.

  Returns:
      float: Scaled Euclidean distance from current to target.
  """
  scale = reduction_factor(index, total_steps)
  distance = np.linalg.norm(current_position - target_position)
  return scale * distance


# Glideslope Algorithm
def multiAgentGlideslope(
    InitNeighbors,
    Nghradius,
    R_0,
    Rdot0,
    R_T,
    R,
    e,
    h,
    omega,
    numJumps,
    dt,
    nframes,
    auctionParams=(True, "Distributed", False),
    Config=([], []),
):
  # Position Initializations
  numAgents = len(R_0)
  R_k = R_0.copy()
  Rdot = Rdot0.copy()
  a_i = [0.00055 for agent in range(numAgents + 1)]
  xs, ys, zs = [[[] for agent in range(numAgents)] for dim in range(3)]
  u_GLS = [(R_0[agent] - R_T[agent]) for agent in range(numAgents)]
  rho_T = [np.linalg.norm(u) for u in u_GLS]
  for agent in range(numAgents):
    if np.linalg.norm(u_GLS[agent]) > 0.0001:
      u_GLS[agent] /= rho_T[agent]
  # Timing Definitions
  T = dt * (nframes - 1)
  deltaT = T / numJumps
  # Energy Initializations
  deltavs = []
  energy = 0.0
  # Neighbors Initializations
  auction = auctionParams[0]
  aucType = auctionParams[1]
  conf = auctionParams[2]
  NeighborsDistanceOT = []
  # Finding Neighbors over Configuration
  if conf:
    NeighborsConfigOT = []
    config = Config[0]
    fConfig = Config[1]
  for firing in range(numJumps):
    # Orbit Parameter Update
    t = deltaT * firing
    tnext = t + deltaT
    f, fnext = time_to_true_anomaly(omega, e, [t, tnext])
    phi_r = getTransformMatrixTH(R, e, h, [tnext, t], [fnext, f])[0:3]
    phi_r_r = [arr[0:3] for arr in phi_r]
    phi_r_rdot = [arr[3:6] for arr in phi_r]
    # Distance-based Auction for Determination of Pre-Assembly Position
    if auction:
      NeighborsDistance = neighborsDistance(InitNeighbors, Nghradius, R_0)
      if aucType == "Standard":
        R_T = stdAuction(R_k, R_T, NeighborsDistance)
      elif aucType == "Hybrid":
        R_T = hybAuction(R_k, R_T, NeighborsDistance)
      elif aucType == "Distributed":
        R_T = disAuction(R_k, R_T, NeighborsDistance)
      elif aucType == "Consensus":
        R_T = cnsAuction(R_k, R_T, NeighborsDistance)
      elif aucType == "Greedy":
        R_T = greedy(R_k, R_T, NeighborsDistance)
    # Configuration-based Auction for Determination of Communication Links
    if conf:
      NeighborsConfig = neighborsConfig(config, fConfig)
    # Individual Agent Update
    for agent in range(numAgents):
      # Check if neighbor is not ego
      if np.linalg.norm(u_GLS[agent]) > 0.0001:
        # Positional Parameter Initialization
        r_T = R_T[agent]
        r_k = R_k[agent]
        r_0 = R_0[agent]
        rdot = Rdot[agent]
        rho_k = np.linalg.norm(r_k - r_T)
        r_k1 = (
            r_T
            + scaled_distance_reduction(firing, numJumps, r_T, r_k, r_0)
            * u_GLS[agent]
        )
        # Delta-V Calculation
        deltav = (
            np.dot(np.linalg.inv(phi_r_rdot), (r_k1 - np.dot(phi_r_r, r_k)))
            - rdot
        )
        deltavs.append(np.linalg.norm(deltav))
        energy += deltavs[-1]
        rdot = rdot + deltav
        # Timed Update of Positional Parameters
        for i in range(int(nframes / numJumps)):
          time = dt * i
          rt, rdott = getManeuverPointsTH(r_k, rdot, [t + time, t], R, e, h)
          xs[agent].append(rt[0])
          ys[agent].append(rt[1])
          zs[agent].append(rt[2])
        # Positional Parameter Accumulation
        R_k[agent] = rt
        Rdot[agent] = rdott
      # If neighbor is ego, do nothing
      else:
        r_0 = R_0[agent]
        xs[agent].append(r_0[0])
        ys[agent].append(r_0[1])
        zs[agent].append(r_0[2])
    # Update Over-Time Collection of Neighbors
    if aucType != "":
      NeighborsDistanceOT.append((deltaT * firing, NeighborsDistance))
    if conf:
      NeighborsConfigOT.append((deltaT * firing, NeighborsConfig))
  # Return Output
  NeighborsOutput = []
  if aucType != "":
    NeighborsOutput.append(NeighborsDistanceOT)
  else:
    NeighborsOutput.append([])
  if conf:
    NeighborsOutput.append(NeighborsConfigOT)
  else:
    NeighborsOutput.append([])
  return xs, ys, zs, deltavs, energy, NeighborsOutput
