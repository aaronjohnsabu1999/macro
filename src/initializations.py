from numpy        import pi, multiply
from numpy.random import rand, seed, randint
seed(0)

def randomNum():
  return ((-1)**randint(0, 2))*rand()

## Random StartPose Assignment
def poseInit(numAgents, radius):
  R_0, Theta = [], []
  for agent in range(numAgents):
    R_0.append  (multiply([randomNum(), randomNum(), randomNum()], radius))
    Theta.append([multiply([randomNum(), randomNum(), randomNum()], pi)])
  Theta.append([[0.0, 0.0, 0.0]])
  return R_0, Theta