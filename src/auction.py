from numpy          import max, sum, zeros, argmax, subtract, random
from numpy.linalg   import norm, solve
from scipy.optimize import linear_sum_assignment

## Generate cost of moving from a particular location to a list of target locations
def costGenerator(egoLocation, targetLocations, minValue):
  return [(-1**(minValue-1))*norm(subtract(egoLocation, targetLocations[i])) for i in range(len(targetLocations))]

## Check connectivity of graph
def isConnected(G):
  numAgents = len(G)
  metAgents = set()
  toVisit   = set()
  visited   = set()
  allAgents = range(len(G))
  metAgents.add(0)
  toVisit.add(0)
  while len(toVisit) != 0:
    ego = list(toVisit)[0]
    if ego not in visited:
      visited.add(ego)
      metAgents.add(ego)
      toVisit.remove(ego)
      toVisit.update([agent for agent in range(numAgents) if G[ego][agent] > 0 and agent not in visited])
  if len(metAgents) == numAgents:
    return True
  return False

## Standard Distributed Auction
def stdAuction(R_0, R_T, G):
  numAgents = len(R_0)
  for ego in range(numAgents):
    R_Tn = [R_T[agent] for agent in range(numAgents) if G[ego][agent] > 0]
    R_0n = [R_0[agent] for agent in range(numAgents) if G[ego][agent] > 0]
    R_TnRange = [agent for agent in range(numAgents) if G[ego][agent] > 0]
    numNbrs = len(R_Tn)
    Auction_Values = [[] for i in range(numNbrs)]
    for nbr, r_0n in enumerate(R_0n):
      Auction_Values[nbr] = costGenerator(r_0n, R_Tn, True)
    row_ind, col_ind = linear_sum_assignment(Auction_Values) # Hungarian Algorithm
    newR_Tn = []
    for i in range(numNbrs):
      newR_Tn.append(R_Tn[col_ind[i]])
    for newAgent, oldAgent in enumerate(R_TnRange):
      R_T[oldAgent] = newR_Tn[newAgent]
  return R_T

## Distributed Greedy Algorithm
def checkIntersections(ego, R_0, R_T, G_ego):
  intNeighbors = []
  r_0 = R_0[ego]
  r_T = R_T[ego]
  for agent, r_0n in enumerate(R_0):
    if G_ego[agent] > 0 and agent != ego:
      r_Tn = R_T[agent]
      A = [[r_0[0]-r_T[0], -(r_0n[0]-r_Tn[0])],
           [r_0[1]-r_T[1], -(r_0n[1]-r_Tn[1])]]
      B = [r_T[0]-r_Tn[0],
           r_T[1]-r_Tn[1]]
      t, s = solve(A, B)
      if abs(((r_0[2]-r_T[2])*t - (r_0n[2]-r_Tn[2])*s) - (r_T[2]-r_Tn[2])) < 0.001:
        intNeighbors.append(agent)
  check = bool(intNeighbors)
  return check, intNeighbors

def greedy(R_0, R_T, G):
  numAgents = len(R_0)
  while True:
    intNeighbors, intersecting = [[] for agent in range(numAgents)], [False for agent in range(numAgents)]
    for ego in range(numAgents):
      intersecting[ego], intNeighbors[ego] = checkIntersections(ego, R_0, R_T, G[ego])
    if not bool(sum(intersecting)):
      break
    for ego in range(numAgents):
      if intersecting[ego]:
        for agent in intNeighbors[ego]:
          R_T[ego], R_T[agent] = R_T[agent], R_T[ego]
          intersecting[ego]  , intNeighbors[ego]   = checkIntersections(ego,   R_0, R_T, G[ego]  )
          intersecting[agent], intNeighbors[agent] = checkIntersections(agent, R_0, R_T, G[agent])
          if not intersecting[ego]:
            break
  return R_T

### 'A Distributed Auction Algorithm for the Assignment Problem' - Zavlanos, Spesivtsev, Pappas
def disAuction(R_0, R_T, G):
  if not isConnected(G):
    return R_T
  
  numAgents = len(R_0)
  beta   = [costGenerator(agent, R_T, False) for agent in R_0]
  p_prev = zeros((numAgents, numAgents))
  p_next = zeros((numAgents, numAgents))
  b_prev = zeros((numAgents, numAgents))
  b_next = zeros((numAgents, numAgents))
  a_prev = zeros( numAgents, dtype = int)
  a_next = zeros( numAgents, dtype = int)
  while len(a_prev) != len(set(a_prev)):
    for ego in range(numAgents):
      for j in range(numAgents):
        p_next[ego][j] = max([p_prev[k][j] for k in range(numAgents) if G[ego][k] > 0])
        b_next[ego][j] = max([b_prev[k][j] for k in range(numAgents) if G[ego][k] > 0 and p_prev[k][j] == p_next[ego][j]])
      if p_prev[ego][a_prev[ego]] <= p_next[ego][a_prev[ego]] and b_next[ego][a_prev[ego]] != ego:
        a_next[ego] = argmax([(beta[ego][k]-p_next[ego][k]) for k in range(numAgents)])
        b_next[ego][a_next[ego]] = ego
        v_ego     = max([beta[ego][k]-p_prev[ego][k] for k in range(numAgents)])
        w_ego     = max([beta[ego][k]-p_prev[ego][k] for k in range(numAgents) if k != a_next[ego]])
        epsilon   = random.rand()*0.05        ####  epsilon-complementary slackness
        gamma_ego = v_ego - w_ego + epsilon
        p_next[ego][a_next[ego]] = p_prev[ego][a_next[ego]] + gamma_ego
      else:
        a_next[ego] = a_prev[ego]
    a_prev = a_next
    p_prev = p_next
    b_prev = b_next
  R_T_new = [R_T[a_ego] for a_ego in a_prev]
  return R_T_new

## 'Consensus-Based Decentralized Auctions for Robust Task Allocation' - Choi, Brunet, How
def selectTask(x_prev, y_prev, J_ego, c_ego, w_ego):
  x_next = x_prev
  y_next = y_prev
  if sum(x_next) == 0:
    h_ego = (c_ego >= y_next)
    if any(h_ego):
      J_ego         = argmax([h_ego[k]*c_ego[k] for k in range(len(y_next))])
      x_next[J_ego] = 1
      y_next[J_ego] = c_ego[J_ego]
  return x_next, y_next, J_ego, w_ego

def updateTask(ego, x_prev, y, z_ego, J_ego, G_ego):
  x_ego = x_prev
  y_ego = y[ego]
  for j in range(len(y)):
    y_ego[j]   = max   ([y[k][j] for k in range(len(y)) if G_ego[k] == 1])
  z_ego[J_ego] = argmax([y[k][J_ego] for k in range(len(y)) if G_ego[k] == 1])
  if not z_ego[J_ego] == ego:
    x_ego[J_ego] = 0
  return x_ego, y_ego, z_ego

def cnsAuction(R_0, R_T, G):
  if not isConnected(G):
    return R_T
  
  numAgents = len(R_0)
  c = [costGenerator(agent, R_T, True) for agent in R_0]
  x = zeros((numAgents, numAgents))
  y = zeros((numAgents, numAgents))
  t = zeros((numAgents, numAgents))
  z = zeros((numAgents, numAgents))
  w = zeros((numAgents, numAgents))
  J = [0 for i in range(numAgents)]
  while 0 in [(1 in x_agent) for x_agent in x]:
    for ego in range(numAgents):
      x[ego], y[ego], J[ego], w[ego] = selectTask(x[ego], y[ego], J[ego], c[ego], w[ego])
    for ego in range(numAgents):
      x[ego], y[ego], z[ego] = updateTask(ego, x[ego], y, z[ego], J[ego], G[ego])
    print(z)
  return R_T

def hybAuction(R_0, R_T, G):
  if not isConnected(G):
    return greedy(R_0, R_T, G)
  else:
    return disAuction(R_0, R_T, G)
