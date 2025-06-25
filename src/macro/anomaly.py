from numpy import cos, sin, multiply

# True Anomaly from Time
def getTrueAnomalyFromTime(omega, e, T, calcDerivs = False):
  M    = multiply(omega, T)
  c1   =  (2.0*e   -(e**3.0)/4.0)
  c2   =  (5.0/4)  *(e**2.0)
  c3   =  (13.0/12)*(e**3.0)
  F    = M + c1*sin(M) + c2*sin(2.0*M) + c3*sin(3.0*M)
  if not calcDerivs:
    return F
  cd1  =  (1.0+(e*cos(F)))**2
  cd2  =  (1.0-(e**2.0))  **(3.0/2)
  Fd   = cd1*omega/cd2
  cdd1 = -2.0*e*(Fd**2.0)
  cdd2 =  (1+(e*cos(F)))
  Fdd  = cdd1*sin(F)/cdd2
  return F, Fd, Fdd