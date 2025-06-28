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
from macro.glideslope import multiAgentGlideslope
from macro.utils import init_pose

num_agents = 18
num_jumps = 5

innerRad = 0.4
R_f = [
    [innerRad * 1 * np.sin(np.deg2rad(0)), innerRad * 1 * np.cos(np.deg2rad(0)), 0.000],
    [
        innerRad * 1 * np.sin(np.deg2rad(60)),
        innerRad * 1 * np.cos(np.deg2rad(60)),
        0.000,
    ],
    [
        innerRad * 1 * np.sin(np.deg2rad(120)),
        innerRad * 1 * np.cos(np.deg2rad(120)),
        0.000,
    ],
    [
        innerRad * 1 * np.sin(np.deg2rad(180)),
        innerRad * 1 * np.cos(np.deg2rad(180)),
        0.000,
    ],
    [
        innerRad * 1 * np.sin(np.deg2rad(240)),
        innerRad * 1 * np.cos(np.deg2rad(240)),
        0.000,
    ],
    [
        innerRad * 1 * np.sin(np.deg2rad(300)),
        innerRad * 1 * np.cos(np.deg2rad(300)),
        0.000,
    ],
    [innerRad * 2 * np.sin(np.deg2rad(0)), innerRad * 2 * np.cos(np.deg2rad(0)), 0.000],
    [
        innerRad * 2 * np.sin(np.deg2rad(30)),
        innerRad * 2 * np.cos(np.deg2rad(30)),
        0.000,
    ],
    [
        innerRad * 2 * np.sin(np.deg2rad(60)),
        innerRad * 2 * np.cos(np.deg2rad(60)),
        0.000,
    ],
    [
        innerRad * 2 * np.sin(np.deg2rad(90)),
        innerRad * 2 * np.cos(np.deg2rad(90)),
        0.000,
    ],
    [
        innerRad * 2 * np.sin(np.deg2rad(120)),
        innerRad * 2 * np.cos(np.deg2rad(120)),
        0.000,
    ],
    [
        innerRad * 2 * np.sin(np.deg2rad(150)),
        innerRad * 2 * np.cos(np.deg2rad(150)),
        0.000,
    ],
    [
        innerRad * 2 * np.sin(np.deg2rad(180)),
        innerRad * 2 * np.cos(np.deg2rad(180)),
        0.000,
    ],
    [
        innerRad * 2 * np.sin(np.deg2rad(210)),
        innerRad * 2 * np.cos(np.deg2rad(210)),
        0.000,
    ],
    [
        innerRad * 2 * np.sin(np.deg2rad(240)),
        innerRad * 2 * np.cos(np.deg2rad(240)),
        0.000,
    ],
    [
        innerRad * 2 * np.sin(np.deg2rad(270)),
        innerRad * 2 * np.cos(np.deg2rad(270)),
        0.000,
    ],
    [
        innerRad * 2 * np.sin(np.deg2rad(300)),
        innerRad * 2 * np.cos(np.deg2rad(300)),
        0.000,
    ],
    [
        innerRad * 2 * np.sin(np.deg2rad(330)),
        innerRad * 2 * np.cos(np.deg2rad(330)),
        0.000,
    ],
]
V_0 = [[0.00, 0.00, 0.00] for i in range(num_agents)]
orbital_radius = 6870 + 405
gravpar = 398600.50
ang_vel = np.sqrt(gravpar / orbital_radius**3)
num_frames = 400
dt = 4000.0 / num_frames
eccentricity = 0.05
ang_mom = (orbital_radius**2) * ang_vel
config = [[1, 2, 1, 1, 1, 2], [2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2]]
fConfig = [i for i in range(num_agents)]

Neighbors = [[0 for i in range(num_agents)] for i in range(num_agents)]
Nghradius = 3.00

# RRR, _ = init_pose(num_agents)
# print(RRR)

RRR = [
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

R_0 = RRR.copy()
Neighbors = [[0 for i in range(num_agents)] for i in range(num_agents)]
X, Y, Z, Deltav, Energy, NeighborsOutput = multiAgentGlideslope(
    Neighbors,
    Nghradius,
    R_0,
    V_0,
    R_f,
    orbital_radius,
    eccentricity,
    ang_mom,
    ang_vel,
    num_jumps,
    dt,
    num_frames,
    auctionParams=("Standard", False),
    Config=(config, fConfig),
)
NeighborsDistanceOT = NeighborsOutput[0]
NeighborsConfigOT = NeighborsOutput[1]
simulatePL(
    "08A-Standard.html",
    X,
    Y,
    Z,
    ["Spacecraft " + str(agent) for agent in range(num_agents)],
)

R_0 = RRR.copy()
Neighbors = [[0 for i in range(num_agents)] for i in range(num_agents)]
X, Y, Z, Deltav, Energy, NeighborsOutput = multiAgentGlideslope(
    Neighbors,
    Nghradius,
    R_0,
    V_0,
    R_f,
    orbital_radius,
    eccentricity,
    ang_mom,
    ang_vel,
    num_jumps,
    dt,
    num_frames,
    auctionParams=("Distributed", False),
    Config=(config, fConfig),
)
NeighborsDistanceOT = NeighborsOutput[0]
NeighborsConfigOT = NeighborsOutput[1]
simulatePL(
    "08B-Distributed.html",
    X,
    Y,
    Z,
    ["Spacecraft " + str(agent) for agent in range(num_agents)],
)

# R_0 = RRR.copy()
# Neighbors = [[0 for i in range(num_agents)] for i in range(num_agents)]
# X, Y, Z, Deltav, Energy, NeighborsOutput = multiAgentGlideslope(Neighbors, Nghradius,
# R_0, V_0, R_f, orbital_radius, eccentricity, ang_mom, ang_vel,
# num_jumps, dt, num_frames,
# auctionParams = ('Consensus', False),
# Config        = (config, fConfig))
# NeighborsDistanceOT = NeighborsOutput[0]
# NeighborsConfigOT   = NeighborsOutput[1]
# simulatePL ('08C-Consensus.html', X, Y, Z, ['Spacecraft '+str(agent) for agent in range(num_agents)])

R_0 = RRR.copy()
Neighbors = [[0 for i in range(num_agents)] for i in range(num_agents)]
X, Y, Z, Deltav, Energy, NeighborsOutput = multiAgentGlideslope(
    Neighbors,
    Nghradius,
    R_0,
    V_0,
    R_f,
    orbital_radius,
    eccentricity,
    ang_mom,
    ang_vel,
    num_jumps,
    dt,
    num_frames,
    auctionParams=("Greedy", False),
    Config=(config, fConfig),
)
NeighborsDistanceOT = NeighborsOutput[0]
NeighborsConfigOT = NeighborsOutput[1]
simulatePL(
    "08D-Greedy.html",
    X,
    Y,
    Z,
    ["Spacecraft " + str(agent) for agent in range(num_agents)],
)
