from copy            import deepcopy
from math            import sqrt, sin, cos, radians
from simulate        import simulatePL
from glideslope      import multiAgentGlideslope
from initializations import poseInit

numAgents = 18
numJumps  = 5

innerRad = 0.4
R_T      = [[innerRad*1*sin(radians(  0)), innerRad*1*cos(radians(  0)), 0.000],
            [innerRad*1*sin(radians( 60)), innerRad*1*cos(radians( 60)), 0.000],
            [innerRad*1*sin(radians(120)), innerRad*1*cos(radians(120)), 0.000],
            [innerRad*1*sin(radians(180)), innerRad*1*cos(radians(180)), 0.000],
            [innerRad*1*sin(radians(240)), innerRad*1*cos(radians(240)), 0.000],
            [innerRad*1*sin(radians(300)), innerRad*1*cos(radians(300)), 0.000],
            
            [innerRad*2*sin(radians(  0)), innerRad*2*cos(radians(  0)), 0.000],
            [innerRad*2*sin(radians( 30)), innerRad*2*cos(radians( 30)), 0.000],
            [innerRad*2*sin(radians( 60)), innerRad*2*cos(radians( 60)), 0.000],
            [innerRad*2*sin(radians( 90)), innerRad*2*cos(radians( 90)), 0.000],
            [innerRad*2*sin(radians(120)), innerRad*2*cos(radians(120)), 0.000],
            [innerRad*2*sin(radians(150)), innerRad*2*cos(radians(150)), 0.000],
            [innerRad*2*sin(radians(180)), innerRad*2*cos(radians(180)), 0.000],
            [innerRad*2*sin(radians(210)), innerRad*2*cos(radians(210)), 0.000],
            [innerRad*2*sin(radians(240)), innerRad*2*cos(radians(240)), 0.000],
            [innerRad*2*sin(radians(270)), innerRad*2*cos(radians(270)), 0.000],
            [innerRad*2*sin(radians(300)), innerRad*2*cos(radians(300)), 0.000],
            [innerRad*2*sin(radians(330)), innerRad*2*cos(radians(330)), 0.000],]
Rdot0    = [[0.00, 0.00, 0.00] for i in range(numAgents)]
R        = 6870 + 405
mu       = 398600.50
omega    = sqrt(mu/R**3)
nframes  =   400
dt       =  4000.0/nframes
e        =  0.05
h        = (R**2)*omega
config   = [[1, 2, 1, 1, 1, 2],
            [2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2]]
fConfig  = [i for i in range(numAgents)]

Neighbors = [[0 for i in range(numAgents)] for i in range(numAgents)]
Nghradius = 3.00

# RRR, _ = poseInit(numAgents)
# print(RRR)

RRR = [[ 2.29364301,  1.26911104, -1.71273596],
       [-2.30499439,  1.46048986,  0.84474138],
       [-2.20553178,  0.33463571, -0.45545043],
       [-2.00737466,  2.69883730,  2.20860278],
       [-2.13433824, -2.45255776,  0.01869546],
       [-2.87274879, -1.20121056, -2.55504595],
       [-2.82212589, -2.88499068, -2.82692798],
       [-2.32413586,  1.38993451, -1.85659537],
       [-2.79356145, -1.84074991,  1.31134734],
       [-2.45089991,  2.86245750, -0.92419686],
       [-2.15760877, -1.63576172, -0.10413470],
       [-2.22633934,  2.55351914, -2.75085611],
       [-2.19934209, -1.28014131,  1.85593947],
       [-2.72048638,  1.75400878, -1.49284592],
       [-2.66459536,  0.56970289, -1.64283741],
       [-2.26072675,  1.64575550, -1.69458091],
       [-2.34255969,  2.06506644, -1.14374557],
       [-2.99163571, -0.24707069,  1.72698837]]

R_0 = deepcopy(RRR)
Neighbors = [[0 for i in range(numAgents)] for i in range(numAgents)]
X, Y, Z, Deltav, Energy, NeighborsOutput = multiAgentGlideslope(Neighbors, Nghradius,
                                                                R_0, Rdot0, R_T, R, e, h, omega,
                                                                numJumps, dt, nframes,
                                                                auctionParams = (True, 'Standard', False),
                                                                Config        = (config, fConfig))
NeighborsDistanceOT = NeighborsOutput[0]
NeighborsConfigOT   = NeighborsOutput[1]
simulatePL ('08A-Standard.html', X, Y, Z, ['Spacecraft '+str(agent) for agent in range(numAgents)])

R_0 = deepcopy(RRR)
Neighbors = [[0 for i in range(numAgents)] for i in range(numAgents)]
X, Y, Z, Deltav, Energy, NeighborsOutput = multiAgentGlideslope(Neighbors, Nghradius,
                                                                R_0, Rdot0, R_T, R, e, h, omega,
                                                                numJumps, dt, nframes,
                                                                auctionParams = (True, 'Distributed', False),
                                                                Config        = (config, fConfig))
NeighborsDistanceOT = NeighborsOutput[0]
NeighborsConfigOT   = NeighborsOutput[1]
simulatePL ('08B-Distributed.html', X, Y, Z, ['Spacecraft '+str(agent) for agent in range(numAgents)])

# R_0 = deepcopy(RRR)
# Neighbors = [[0 for i in range(numAgents)] for i in range(numAgents)]
# X, Y, Z, Deltav, Energy, NeighborsOutput = multiAgentGlideslope(Neighbors, Nghradius,
                                                                # R_0, Rdot0, R_T, R, e, h, omega,
                                                                # numJumps, dt, nframes,
                                                                # auctionParams = (True, 'Consensus', False),
                                                                # Config        = (config, fConfig))
# NeighborsDistanceOT = NeighborsOutput[0]
# NeighborsConfigOT   = NeighborsOutput[1]
# simulatePL ('08C-Consensus.html', X, Y, Z, ['Spacecraft '+str(agent) for agent in range(numAgents)])

R_0 = deepcopy(RRR)
Neighbors = [[0 for i in range(numAgents)] for i in range(numAgents)]
X, Y, Z, Deltav, Energy, NeighborsOutput = multiAgentGlideslope(Neighbors, Nghradius,
                                                                R_0, Rdot0, R_T, R, e, h, omega,
                                                                numJumps, dt, nframes,
                                                                auctionParams = (True, 'Greedy', False),
                                                                Config        = (config, fConfig))
NeighborsDistanceOT = NeighborsOutput[0]
NeighborsConfigOT   = NeighborsOutput[1]
simulatePL ('08D-Greedy.html', X, Y, Z, ['Spacecraft '+str(agent) for agent in range(numAgents)])