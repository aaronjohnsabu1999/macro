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

from macro.ifmea import listPointToTwoDimPoint, twoDimPointToListPoint


# Determination of Neighbors based on Configuration
def neighborsConfig(config, fConfig):
    LayerLens = [len(layer) for layer in config]
    numAgents = len(fConfig)

    N = [[] for i in range(numAgents)]
    for i in range(numAgents):
        presentLayerNum, presentPositionNum = listPointToTwoDimPoint(LayerLens, i)
        N[i] = [
            fConfig[
                twoDimPointToListPoint(
                    LayerLens,
                    presentLayerNum,
                    (presentPositionNum - 1) % LayerLens[presentLayerNum],
                )
            ],
            fConfig[
                twoDimPointToListPoint(
                    LayerLens,
                    presentLayerNum,
                    (presentPositionNum + 1) % LayerLens[presentLayerNum],
                )
            ],
        ]
        if presentLayerNum != len(LayerLens) - 1:
            N[i].append(
                fConfig[
                    twoDimPointToListPoint(
                        LayerLens, presentLayerNum + 1, (presentPositionNum * 2) - 1
                    )
                ]
            )
            N[i].append(
                fConfig[
                    twoDimPointToListPoint(
                        LayerLens, presentLayerNum + 1, (presentPositionNum * 2) + 0
                    )
                ]
            )
            N[i].append(
                fConfig[
                    twoDimPointToListPoint(
                        LayerLens, presentLayerNum + 1, (presentPositionNum * 2) + 1
                    )
                ]
            )
        if presentLayerNum != 0:
            if presentPositionNum % 2:
                N[i].append(
                    fConfig[
                        twoDimPointToListPoint(
                            LayerLens,
                            presentLayerNum - 1,
                            int((presentPositionNum - 1) / 2),
                        )
                    ]
                )
                N[i].append(
                    fConfig[
                        twoDimPointToListPoint(
                            LayerLens,
                            presentLayerNum - 1,
                            int((presentPositionNum + 1) / 2),
                        )
                    ]
                )
            else:
                N[i].append(
                    fConfig[
                        twoDimPointToListPoint(
                            LayerLens,
                            presentLayerNum - 1,
                            int((presentPositionNum + 0) / 2),
                        )
                    ]
                )
        else:
            N[i].append(numAgents)
    return N


# Determination of Neighbors based on Distance
def neighborsDistance(Neighbors, Nghradius, R_0):
    for egoCount, r_0 in enumerate(R_0):
        for j, r_1 in enumerate(R_0[egoCount:]):
            nghCount = egoCount + j
            if np.linalg.norm(r_0 - r_1) < Nghradius:  # and not(j == egoCount):
                value = 1
            else:
                value = 0
            Neighbors[egoCount][nghCount] = value
            Neighbors[nghCount][egoCount] = value
    return Neighbors
