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

import math
import matplotlib.pyplot as mplplt
import plotly.offline as ptyplt
import plotly.graph_objects as go
from random import randint
from matplotlib import animation

mplplt.style.use("fivethirtyeight")


# Simulations
def simulateMPL(title, X, Y, Z, labels, nframes):  # Simulate using Matplotlib
  def init():
    return ax

  def animate(frame):
    plots = []
    for agent in range(numAgents):
      plots.append(
          ax.plot(
              X[agent][frame],
              Y[agent][frame],
              Z[agent][frame],
              marker=".",
              label=labels[agent],
          )
      )
    return plots

  numAgents = len(X)
  xlim = (min(X[0]), max(X[0]))
  ylim = (min(Y[0]), max(Y[0]))
  zlim = (min(Z[0]), max(Z[0]))
  for agent in range(numAgents):
    xlim = (min(xlim[0], min(X[agent])), max(xlim[1], max(X[agent])))
    ylim = (min(ylim[0], min(Y[agent])), max(ylim[1], max(Y[agent])))
    zlim = (min(zlim[0], min(Z[agent])), max(zlim[1], max(Z[agent])))
  fig = mplplt.figure()
  ax = fig.add_subplot(111, projection="3d", xlim=xlim, ylim=ylim, zlim=zlim)
  ax.legend(loc="upper right", shadow=True)
  ax.set_xlabel("X")
  ax.set_ylabel("Y")
  ax.set_zlabel("Z")
  ax.azim = 140
  ax.elev = -45
  anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nframes)
  anim.save(title, fps=20, writer="Pillow",
            progress_callback=lambda i, n: print(i))


def simulatePL(title, X, Y, Z, labels):  # Simulate using Plotly
  markers = []
  for agent in range(len(X)):
    markers.append(
        go.Scatter3d(
            mode="markers",
            x=X[agent],
            y=Y[agent],
            z=Z[agent],
            marker=dict(size=3),
            name=labels[agent],
        )
    )
  """
  numSplits = 5
  for agent in range(len(X)):
    dist  = len(X[agent])
    color = randint(0,255)
    for point in range(numSplits):
      size = 5.0*math.exp(-point/numSplits)
      markers.append(go.Scatter3d(mode = 'markers', name = labels[agent],
                                  x = [X[agent][i] for i in range(int(point*dist/numSplits), int((point+1)*dist/numSplits))],
                                  y = [Y[agent][i] for i in range(int(point*dist/numSplits), int((point+1)*dist/numSplits))],
                                  z = [Z[agent][i] for i in range(int(point*dist/numSplits), int((point+1)*dist/numSplits))],
                                  marker = dict(size = size, color = color)))
  """
  fig = go.Figure(data=markers)
  ptyplt.plot(fig, filename=title)


def plotRPY(Theta):
  numAgents = len(Theta) - 1
  for dim in range(3):
    for agent in range(numAgents):
      mplplt.plot([Theta[agent][i][dim] for i in range(len(Theta[agent]))])
    mplplt.show()
