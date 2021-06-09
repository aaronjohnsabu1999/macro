from copy         import deepcopy
from numpy        import add, dot, subtract, multiply
from auction      import greedy, cnsAuction, disAuction, hybAuction, stdAuction
from anomaly      import getTrueAnomalyFromTime
from maneuvers    import getTransformMatrixTH, getManeuverPointsTH
from neighbors    import neighborsConfig, neighborsDistance
from numpy.linalg import inv, norm

## Reduction Functions
def reductionF(x, N):
  return 1.0 - (x+1)/N

def reductionG(x, N, r_T, r_k, r_0):
  return reductionF(x, N)*norm(subtract(r_k, r_T))

## Glideslope Algorithm
def multiAgentGlideslope(InitNeighbors, Nghradius, R_0, Rdot0, R_T, R, e, h, omega, numJumps, dt, nframes, auctionParams = (True, 'Distributed', False), Config = ([], [])):
  ### Position Initializations
  numAgents  = len(R_0)
  R_k        = deepcopy(R_0)
  Rdot       = deepcopy(Rdot0)
  a_i        = [0.00055 for agent in range(numAgents+1)]
  xs, ys, zs = [ [[] for agent in range(numAgents)] for dim in range(3) ]
  u_GLS      = [subtract(R_0[agent], R_T[agent]) for agent in range(numAgents)]
  rho_T      = [norm(u) for u in u_GLS]
  for agent in range(numAgents):
    if norm(u_GLS[agent]) > 0.0001:
      u_GLS[agent] /= rho_T[agent]
  ### Timing Definitions
  T       = dt*(nframes-1)
  deltaT  = T/numJumps
  ### Energy Initializations
  deltavs = []
  energy  = 0.0
  ### Neighbors Initializations
  auction = auctionParams[0]
  aucType = auctionParams[1]
  conf    = auctionParams[2]
  NeighborsDistanceOT = []
  ### Finding Neighbors over Configuration
  if conf:
    NeighborsConfigOT   = []
    config              = Config[0]
    fConfig             = Config[1]
  for firing in range(numJumps):
    ### Orbit Parameter Update
    t          = deltaT*firing
    tnext      = t + deltaT
    f, fnext   = getTrueAnomalyFromTime(omega, e, [t, tnext])
    phi_r      = getTransformMatrixTH(R, e, h, [tnext, t], [fnext, f])[0:3]
    phi_r_r    = [arr[0:3] for arr in phi_r]
    phi_r_rdot = [arr[3:6] for arr in phi_r]
    ### Distance-based Auction for Determination of Pre-Assembly Position
    if auction:
      NeighborsDistance   = neighborsDistance(InitNeighbors, Nghradius, R_0)
      if   aucType == 'Standard':
        R_T = stdAuction(R_k, R_T, NeighborsDistance)
      elif aucType == 'Hybrid':
        R_T = hybAuction(R_k, R_T, NeighborsDistance)
      elif aucType == 'Distributed':
        R_T = disAuction(R_k, R_T, NeighborsDistance)
      elif aucType == 'Consensus':
        R_T = cnsAuction(R_k, R_T, NeighborsDistance)
      elif aucType == 'Greedy':
        R_T = greedy    (R_k, R_T, NeighborsDistance)
    ### Configuration-based Auction for Determination of Communication Links
    if conf:
      NeighborsConfig = neighborsConfig(config, fConfig)
    ### Individual Agent Update
    for agent in range(numAgents):
      ### Check if neighbor is not ego
      if norm(u_GLS[agent]) > 0.0001:
        ### Positional Parameter Initialization
        r_T   = R_T[agent]
        r_k   = R_k[agent]
        r_0   = R_0[agent]
        rdot  = Rdot[agent]
        rho_k = norm(subtract(r_k, r_T))
        r_k1  = add(r_T, multiply(reductionG(firing, numJumps, r_T, r_k, r_0), u_GLS[agent]))
        ### Delta-V Calculation
        deltav  = subtract(dot(inv(phi_r_rdot), (r_k1 - dot(phi_r_r, r_k))), rdot)
        deltavs.append(norm(deltav))
        energy += deltavs[-1]
        rdot    = add(rdot, deltav)
        ### Timed Update of Positional Parameters
        for i in range(int(nframes/numJumps)):
          time      = dt*i
          rt, rdott = getManeuverPointsTH(r_k, rdot, [t + time, t], R, e, h)
          xs[agent].append(rt[0])
          ys[agent].append(rt[1])
          zs[agent].append(rt[2])
        ### Positional Parameter Accumulation
        R_k [agent] = rt
        Rdot[agent] = rdott
      ### If neighbor is ego, do nothing
      else:
        r_0 = R_0[agent]
        xs[agent].append(r_0[0])
        ys[agent].append(r_0[1])
        zs[agent].append(r_0[2])
    ### Update Over-Time Collection of Neighbors
    if aucType != '':
      NeighborsDistanceOT.append((deltaT*firing, NeighborsDistance))
    if conf:
      NeighborsConfigOT.append((deltaT*firing, NeighborsConfig))
  ### Return Output
  NeighborsOutput = []
  if aucType != '':
    NeighborsOutput.append(NeighborsDistanceOT)
  else:
    NeighborsOutput.append([])
  if conf:
    NeighborsOutput.append(NeighborsConfigOT)
  else:
    NeighborsOutput.append([])
  return xs, ys, zs, deltavs, energy, NeighborsOutput