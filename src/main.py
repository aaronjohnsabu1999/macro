from copy            import deepcopy
from math            import sqrt
from ifmea           import ifmea, listPointToTwoDimPoint
from numpy           import pi, argmin, random, subtract
from mirror          import genR_T, layerCalc, calcTheta_d
from simulate        import plotRPY, simulatePL, simulateMPL
from consensus       import stepUpdate
from glideslope      import multiAgentGlideslope
from numpy.linalg    import norm
from numpy.random    import seed, shuffle
from initializations import poseInit
seed(0)

### Orbit Parameters
R     = 6870 + 405
mu    = 398600.50
omega = sqrt(mu/R**3)
e     =  0.05
h     = (R**2)*omega
### Simulation Parameters
nframes =   800
dt      =  4000.0/nframes
### Configuration Parameters
numAgents     = 36
numJumps      = 10
Nghradius     =  0.4
clearAperture =  0.002
parabolicRad  =  0.003
expansion     =  3
Neighbors     = [[0 for i in range(numAgents)] for i in range(numAgents)]
a             = [[[0.5*4e-6, 0.5*8e-6, 0.5*8e-6] for i in range(numAgents + 1)] for j in range(numAgents + 1)]
numLayers, LayerLens = layerCalc(numAgents)
### Position Parameters
R_T0  = genR_T(clearAperture, parabolicRad, numAgents, LayerLens, expansion)
R_T   = deepcopy(R_T0)
Rdot0 = [[0.00, 0.00, 0.00] for i in range(numAgents)]
R_0, Theta  = poseInit(numAgents, 5.00)
R_I, _      = poseInit(numAgents, 0.01)
individuals = [i for i in range(numAgents)]
trueMapping = []
for layer in range(1, numLayers + 1):
  trueMapping.extend([layer for i in range(LayerLens[layer-1])])
# shuffle(trueMapping)

### Glideslope with Hybrid Auction at jumps from Start Positions to Intermediate Positions
Xs, Ys, Zs, Deltav, Energy, NeighborsOutput = multiAgentGlideslope(Neighbors, Nghradius,
                                                                   R_0, Rdot0, R_I, R, e, h, omega,
                                                                   numJumps, dt, nframes,
                                                                   auctionParams = (True, 'Hybrid', False),
                                                                   Config = (trueMapping, individuals))
NeighborsDistanceOT = NeighborsOutput[0]
NeighborsConfigOT   = NeighborsOutput[1]
simulatePL('Stage-01A.html', Xs, Ys, Zs,
          ['S/c '+str(agent) for agent in range(numAgents)])
# simulateMPL('Stage-01A.gif', Xs, Ys, Zs,
           # ['S/c '+str(agent) for agent in range(numAgents)], nframes)

### Glideslope with Hybrid Auction at jumps from Intermediate Positions to R_T
Xs, Ys, Zs, Deltav, Energy, NeighborsOutput = multiAgentGlideslope(Neighbors, Nghradius,
                                                                   R_I, Rdot0, R_T, R, e, h, omega,
                                                                   numJumps, dt, nframes,
                                                                   auctionParams = (True, 'Hybrid', False),
                                                                   Config = (trueMapping, individuals))
NeighborsDistanceOT = NeighborsOutput[0]
NeighborsConfigOT   = NeighborsOutput[1]
simulatePL('Stage-01B.html', Xs, Ys, Zs,
          ['S/c '+str(agent) for agent in range(numAgents)])
# simulateMPL('Stage-01B.gif', Xs, Ys, Zs,
           # ['S/c '+str(agent) for agent in range(numAgents)], nframes)

### Determine present configuration of spacecraft
presentConfig = [[] for layer in range(len(LayerLens))]
for agent in range(numAgents):
  r_0 = [Xs[agent][-1], Ys[agent][-1], Zs[agent][-1]]
  presLayerNum, _ = listPointToTwoDimPoint(LayerLens, argmin([norm(subtract(r_0, r_T)) for r_T in R_T]))
  trueLayer = trueMapping[agent]
  presentConfig[presLayerNum].append(trueLayer)

### Intra-Formation Mutual Exchange
Xs, Ys, Zs, Rolls, Pitches, Yaws = [[[] for agent in range(numAgents)] for i in range(6)]
AllNeighborsOverTime, FConfig    = [ [] for     i in range(2)]
IFMEACommands = ifmea(presentConfig)
R_0 = R_T
for step, command in enumerate(IFMEACommands):
  for i in range(command[4]):
    if command[0] == 'R':
      rotInd  = individuals[command[1]:command[2]+1]
      rotInd  = rotInd[-command[3]:] + rotInd[:-command[3]]
      indTemp = deepcopy(individuals[:command[1]])
      indTemp.extend(rotInd)
      indTemp.extend(individuals[command[2]+1:])
      individuals = indTemp
    elif command[0] == 'E':
      R_T = deepcopy(R_0)
      ind1, ind2, ind3 = individuals[command[1]], individuals[command[2]], individuals[command[3]]
      R_T[ind1], R_T[ind2], R_T[ind3] = R_T[ind2], R_T[ind3], R_T[ind1]
      individuals[command[1]], individuals[command[2]], individuals[command[3]] = ind3, ind1, ind2
    FConfig.append(individuals)
    for agent in individuals:
      R_T[agent] = R_T0[individuals.index(agent)]
    X, Y, Z, Deltav, Energy, NeighborsOutput = multiAgentGlideslope(Neighbors, Nghradius,
                                                                    R_0, Rdot0, R_T, R, e, h, omega,
                                                                    numJumps, dt, nframes,
                                                                    auctionParams = (False, '', True),
                                                                    Config = (presentConfig, individuals))
    NeighborsDistanceOT = NeighborsOutput[0]
    NeighborsConfigOT   = NeighborsOutput[1]
    AllNeighborsOverTime.append(NeighborsConfigOT)
    R_0 = R_T
    for agent in range(numAgents):
      Xs[agent].extend(X[agent])
      Ys[agent].extend(Y[agent])
      Zs[agent].extend(Z[agent])
simulatePL('Stage-02.html', Xs, Ys, Zs,
          ['S/c '+str(agent) for agent in range(numAgents)])

### Attitude Consensus
for command in range(len(AllNeighborsOverTime)):
  time    = 0
  fConfig = FConfig[command]
  NeighborsConfigOT = AllNeighborsOverTime[command]
  Theta_d = calcTheta_d(clearAperture, parabolicRad, LayerLens, FConfig[command])
  for i in range(nframes):
    Neighbors = NeighborsConfigOT[0][1]
    tempTheta = []
    for agent in range(numAgents):
      tempTheta.append(stepUpdate(agent, Theta, Theta_d, Neighbors[agent], a[agent], dt, i))
    tempTheta.append([0.0, 0.0, 0.0])
    for agent in range(numAgents):
      Theta[agent].append(tempTheta[agent])
plotRPY(Theta)