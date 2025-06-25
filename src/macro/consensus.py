from numpy import subtract, multiply

## Attitude Consensus
def stepUpdate(ego, Theta, Theta_d, N_ego, a_ego, step, iteration):
  nextTheta_ego = Theta[ego][-1]
  for nbr, a_ego_nbr in enumerate(a_ego):
    if nbr in N_ego:
      nextTheta_ego = subtract(nextTheta_ego, step * iteration * multiply(a_ego_nbr, subtract(subtract(Theta[ego][-1], Theta_d[ego]), subtract(Theta[nbr][-1], Theta_d[nbr]))))
  return nextTheta_ego