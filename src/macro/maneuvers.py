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

from macro.utils import load_config, time_to_true_anomaly

config = load_config()
gravpar = config["orbit"]["gravpar"]


# Transform Matrices
def getTransformMatrixCWH(O, t):
  OT = O * t
  cOT = np.cos(OT)
  sOT = np.sin(OT)
  A = [
      [
          1.0,
          0.0,
          6.0 * OT - 6.0 * sOT,
          4.0 * sOT / O - 3.0 * t,
          0.0,
          2.0 * (1.0 - cOT) / O,
      ],
      [0.0, cOT, 0.0, 0.0, sOT / O, 0.0],
      [0.0, 0.0, 4.0 - 3.0 * cOT, 2.0 * (cOT - 1.0) / O, 0.0, sOT / O],
      [0.0, 0.0, 6 * O * (1.0 - cOT), 4.0 * cOT - 3.0, 0.0, 2.0 * sOT],
      [0.0, -O * sOT, 0.0, 0.0, cOT, 0.0],
      [0.0, 0.0, 3.0 * sOT * O, -2.0 * sOT, 0.0, cOT],
  ]
  return A


def getTransformMatrixTH(R, e, h, T, F):
  sf = [np.sin(f) for f in F]
  cf = [np.cos(f) for f in F]
  rho = [1 + e * cf[i] for i in [0, 1]]
  p = [R * rho[i] for i in [0, 1]]
  K1 = 1.0 / (1.0 - (e**2.0))
  K2 = [h / (p[i] ** 2) for i in [0, 1]]
  K3 = [1.0 + (1.0 / rho[i]) for i in [0, 1]]

  J = K2[0] * (T[0] - T[1])
  c = [rho[i] * cf[i] for i in [0, 1]]
  s = [rho[i] * sf[i] for i in [0, 1]]
  cd = [-(sf[i] + e * np.sin(2 * F[i])) for i in [0, 1]]
  sd = [cf[i] + e * np.cos(2 * F[i]) for i in [0, 1]]

  cY = np.cos(F[0] - F[1])
  sY = np.sin(F[0] - F[1])
  rhoY = 1 + e * cY
  pY = R * rhoY
  K2Y = h / pY**2

  T1 = [
      [1.0 / rho[0], 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 1.0 / rhoY, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 1.0 / rho[0], 0.0, 0.0, 0.0],
      [K2[0] * e * sf[0], 0.0, 0.0, rho[0] * K2[0], 0.0, 0.0],
      [0.0, K2Y * e * sY, 0.0, 0.0, rhoY * K2Y, 0.0],
      [0.0, 0.0, K2[0] * e * sf[0], 0.0, 0.0, rho[0] * K2[0]],
  ]

  A1 = [
      [1.0, 0.0, -K3[0] * c[0], K3[0] * s[0], 0.0, 3.0 * J * (rho[0] ** 2)],
      [0.0, cY, 0.0, 0.0, sY, 0.0],
      [0.0, 0.0, s[0], c[0], 0.0, 2.0 - 3.0 * e * s[0] * J],
      [0.0, 0.0, 2.0 * s[0], (2 * c[0] - e), 0.0, 3.0 *
       (1.0 - 2.0 * e * s[0] * J)],
      [0.0, -sY, 0.0, 0.0, cY, 0.0],
      [0.0, 0.0, sd[0], cd[0], 0.0, -3.0 *
       e * (sd[0] * J + s[0] / (rho[0] ** 2))],
  ]

  A2 = [
      [
          1.0,
          0.0,
          3.0 * K1 * K3[1] * e * s[1] / rho[1],
          -K1 * K3[1] * e * s[1],
          0.0,
          K1 * (-e * c[1] + 2.0),
      ],
      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
      [
          0.0,
          0.0,
          -3.0 * K1 * s[1] * (1.0 + (e**2) / rho[1]) / rho[1],
          K1 * K3[1] * s[1],
          0.0,
          K1 * (c[1] - 2.0 * e),
      ],
      [
          0.0,
          0.0,
          -3.0 * K1 * ((c[1] / rho[1]) + e),
          K1 * (K3[1] * c[1] + e),
          0.0,
          -K1 * s[1],
      ],
      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
      [
          0.0,
          0.0,
          K1 * (3.0 * rho[1] + (e**2) - 1.0),
          -K1 * (rho[1] ** 2),
          0.0,
          K1 * e * s[1],
      ],
  ]

  T2 = [
      [rho[1], 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, rhoY, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, rho[1], 0.0, 0.0, 0.0],
      [-e * sf[1], 0.0, 0.0, 1.0 / (rho[1] * K2[1]), 0.0, 0.0],
      [0.0, -e * sY, 0.0, 0.0, 1.0 / (rhoY * K2Y), 0.0],
      [0.0, 0.0, -e * sf[1], 0.0, 0.0, 1.0 / (rho[1] * K2[1])],
  ]

  return np.dot(T1, np.dot(np.dot(A1, A2), T2))


# Maneuver Points
def getManeuverPointsCWH(r0, rdot0, t, omega):
  P = getTransformMatrixCWH(omega, t)
  a0 = [r0[0], r0[1], r0[2], rdot0[0], rdot0[1], rdot0[2]]
  at = np.dot(P, a0)
  return at[0:3], at[3:6]


def getManeuverPointsTH(r0, rdot0, T, R, e, h):
  omega = np.sqrt(gravpar / R**3)
  F = time_to_true_anomaly(omega, e, T)
  P = getTransformMatrixTH(R, e, h, T, F)
  a0 = [r0[0], r0[1], r0[2], rdot0[0], rdot0[1], rdot0[2]]
  at = np.dot(P, a0)
  return at[0:3], at[3:6]
