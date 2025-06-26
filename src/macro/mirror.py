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

from macro.ifmea import listPointToTwoDimPoint


# Calculate lengths of layers for given number of agents
def layerCalc(numAgents):
  layer = 1
  layerLens = []
  while np.sum(layerLens) < numAgents:
    layerLens.append(layer * 6)
    layer += 1
  return len(layerLens), layerLens


# Calculate specific positions for all agents
def genR_T(CA, R, numAgents, LayerLens, expansion):
  R_T, XY, Z = [[] for i in range(3)]
  numLayers = len(LayerLens)
  _, Ps = parabolicParams(CA, R, numLayers)
  for layer in range(numLayers):
    XY.append(Ps[layer][0])
    Z.append(Ps[layer][1])
  for agent in range(numAgents):
    layerNum, pointNum = listPointToTwoDimPoint(LayerLens, agent)
    circAngle = pointNum * 2.0 * np.pi / LayerLens[layerNum]
    R_T.append(
        [
            expansion * XY[layerNum] * np.sin(circAngle),
            expansion * XY[layerNum] * np.cos(circAngle),
            expansion * Z[layerNum],
        ]
    )
  return R_T


# Calculate specific attitudes for all agents
def calcTheta_d(CA, R, LayerLens, fConfig):
  Theta_d = [[0.0, 0.0, 0.0] for j in range(len(fConfig) + 1)]
  Pitches, _ = parabolicParams(CA, R, len(LayerLens))
  for listPoint, agent in enumerate(fConfig):
    presLayerNum, presPointNum = listPointToTwoDimPoint(LayerLens, listPoint)
    trueLayerNum, truePointNum = listPointToTwoDimPoint(LayerLens, agent)
    Theta_d[agent][1] = np.pi / 2.0 - Pitches[trueLayerNum]
    Theta_d[agent][2] = presPointNum * 2.0 * np.pi / LayerLens[presLayerNum]
    if Theta_d[agent][2] > np.pi:
      Theta_d[agent][2] -= 2.0 * np.pi
  return Theta_d


# Generate generic positions and attitudes based on layer number
def parabolicParams(CA, R, l):
  def equations(theta_i, Thetas):
    lhs = CA * (0.5 + np.sum([np.sin(theta)
                for theta in Thetas]) + np.sin(theta_i))
    rhs = CA * (
        CA / (8.0 * R)
        + np.sum([np.cos(theta) for theta in Thetas])
        + np.cos(theta_i)
    )
    return (lhs**2) / (2.0 * R) - rhs

  Thetas = []
  Ps = [(CA / 2.0, CA**2 / (8.0 * R))]
  for i in range(l):
    theta_i = fsolve(equations, 0, Thetas)
    Thetas.append(theta_i[0])
    p_i = (
        CA * (0.5 + np.sum([np.sin(theta) for theta in Thetas])),
        CA * (CA / (8.0 * R) + np.sum([np.cos(theta) for theta in Thetas])),
    )
    Ps.append(p_i)
  return Thetas, Ps
