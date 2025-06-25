from math import floor

def twoDimPointToListPoint(layerLens, presentLayerNum, presentPosNum):
  sumUptoLayer = 0
  for layerLen in layerLens[:presentLayerNum]:
    sumUptoLayer += layerLen
  return sumUptoLayer + (presentPosNum % layerLens[presentLayerNum])

def listPointToTwoDimPoint(layerLens, listPoint):
  for layerNum, layerLen in enumerate(layerLens):
    if listPoint < layerLen:
      return (layerNum, listPoint)
    listPoint -= layerLen

def findNearestMismatch(config, nextConfig, presentLayerNum, presentPosNum):
  n_L_prst = len(config)
  n_L_next = len(nextConfig)
  for dist in range(0, int(n_L_next/2)+1):
    upper = (presentPosNum*2 + dist)%n_L_next
    lower = (presentPosNum*2 - dist)%n_L_next
    if nextConfig[upper] != presentLayerNum + 2:
      nextLayerMismatch = upper
      break
    elif nextConfig[lower] != presentLayerNum + 2:
      nextLayerMismatch = lower
      break
  presLayerMMClosest = floor(nextLayerMismatch/2)
  if presLayerMMClosest < presentPosNum:
    presLayerMMClosest += nextLayerMismatch%2
  return nextLayerMismatch, presLayerMMClosest

def rotateConfig(layer, diff):
  for i in range(abs(diff)):
    layer = layer[-int(diff/abs(diff)):] + layer[:-int(diff/abs(diff))]
  return layer
    
def exchangeConfig(config, P1, P2, P3):
  config[P1[0]][P1[1]], config[P2[0]][P2[1]], config[P3[0]][P3[1]] = config[P3[0]][P3[1]], config[P1[0]][P1[1]], config[P2[0]][P2[1]]
  return config

def ifmea(config):
  N = len(config) - 1
  commands = []
  layerLens = [len(layer) for layer in config]
  while not N == 0:
    i = N - 1
    while not i == -1:
      try:
        j = config[i].index(N+1)
      except:
        i -= 1
        continue
      m, l = findNearestMismatch(config[i], config[i+1], i, j)
      if l-j != 0:
        config[i] = rotateConfig(config[i], l-j)
        commands.append(('R', twoDimPointToListPoint(layerLens, i, 0), twoDimPointToListPoint(layerLens, i+1, 0) - 1, int(l-j/abs(l-j)), abs(l-j)))
      if m < l*2:
        config = exchangeConfig(config, (i,l), (i+1,m), (i,l-1))
        commands.append(('E', twoDimPointToListPoint(layerLens, i, l), twoDimPointToListPoint(layerLens, i+1, m), twoDimPointToListPoint(layerLens, i, l-1), 1))
      else:
        config = exchangeConfig(config, (i,l), (i+1,m), (i,(l+1)%layerLens[i]))
        commands.append(('E', twoDimPointToListPoint(layerLens, i, l), twoDimPointToListPoint(layerLens, i+1, m), twoDimPointToListPoint(layerLens, i, l+1), 1))
      if i < N - 1:
        i = i + 1
    N = N - 1
  
  return commands

if __name__ == '__main__':
  initConfig = [[1,          1,          3,          1,          2,          2],
                [2,    2,    1,    2,    2,    2,    2,    3,    2,    2,    2,    1],
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]
  print(ifmea(initConfig))