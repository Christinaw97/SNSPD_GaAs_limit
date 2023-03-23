import numpy as np
import scipy as sp
import scipy.special
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as clb
import matplotlib.cm as cm
import pdb
import concurrent.futures
import matplotlib.ticker

"""Global Constants & Conversion"""
amu2eV = 9.315e8                                             # eV/u
lightSpeed = 299792.458                                     # In km/s
pi = np.pi
alpha = 1.0/137.03599908                                    # EM fine-structure constant at low E
mElectron = 5.1099894e5                                     # In eV
BohrInv2eV = alpha*mElectron                                # 1 Bohr^-1 to eV
Ryd2eV = 0.5*mElectron*alpha**2                             # In eV/Ryd
cm2sec = 1/lightSpeed*1e-5                                  # 1 cm in s
sec2yr = 1/(60.*60.*24*365.25)                              # 1 s in years
cmInv2eV = 50677.3093773                                    # 1 eV in cm^-1

"""Crystal Parameters"""
atomicMass = np.array([69.723, 74.922])*amu2eV                     # mass of each species in unit cell, input in amu, converted to eV
atomsInCell = np.array([1, 1])                                 # number of atoms of each species, ensure same index as in atomicMass
lat_vecs = np.array([[2.82675, 2.82675, 0. ],\
      [0., 2.82675, 2.82675], [2.82675, 0., 2.82675]])*1e-8         # In cm
PUC_in_Cell = 1.0                                           # Number of primitive unit cell in cell
VCell = np.abs(lat_vecs[0]@\
     np.cross(lat_vecs[1], lat_vecs[2]))\
          *(cmInv2eV**3)/PUC_in_Cell                        # Volume of cell in eV^-3
wk = 2.0/137.0                                              # weight of each k point
MCell = np.sum(atomicMass*atomsInCell)                      # number of atoms in Cell * atomic mass (In eV)
bandGap = 1.42                                              # In eV
MTarget = 1.0*5.609588e35                                   # In eV
exposure = 1.0*31556952.0                                   # In s
Egap = 1.42                                                 # Bandgap in eV
E2q = 2.9                                                   # E to excite 1 electron. 
eps0 = 14.0                                                 # Unitless
qTF = 3.99e3                                                # In eV
omegaP = 15.2                                               # In eV
alphaS = 1.563                                              # Unitless

"""Input parameters"""
pyscfFile = "f2.npy"
QEDout = "C.dat"
QEDark = False                                              # Read QEDark output or pyscf
#QEDark_mod = False                                         # weight calculation needed? By default, no
dq = 0.02                                                   # In Bohr^{-1}
q_shift = 0.0                                               # In Bohr^{-1}
dE = 0.1                                                    # In eV
E_shift = 0.0                                               # In eV
dq, q_shift = dq*BohrInv2eV, q_shift*BohrInv2eV             # Convert dq, dE units to eV
Ebins, startE = int(np.round(E2q/dE)), int(Egap/dE)                   # Use crystal parameters for charge production.

"""Dark Matter Parameters"""
v0 = 230.0/lightSpeed                                       # In units of c
vEarth = 240.0/lightSpeed                                   # In units of c
vEscape = 600.0/lightSpeed                                  # In units of c
rhoX = 0.4e9                                                # In eV/cm^3
crosssection = 1e-39                                        # In cm^2

"""Make Plots Pretty!"""
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

"""Read the f2 file from Cheng"""
def read_f2(fileName):
     f2 = np.load(fileName)
     return f2

"""Reduced Mass"""
def reducedMass(mX):                                        # In eV
     return mX*mElectron/(mX + mElectron)

"""Integrated Maxwell-Boltzmann Distribution"""
def eta_MB(qArr, E, mX):                                    # In units of c^-1
     
     val = list()
     for q in qArr:
          vMin = q/(2.0*mX) + E/q
          if (vMin < vEscape - vEarth):
               val.append(-4.0*vEarth*np.exp(-(vEscape/v0)**2) + np.sqrt(pi)*v0*(sp.special.erf((vMin+vEarth)/v0) - sp.special.erf((vMin - vEarth)/v0)))
          elif (vMin < vEscape + vEarth):
               val.append(-2.0*(vEarth+vEscape-vMin)*np.exp(-(vEscape/v0)**2) + np.sqrt(pi)*v0*(sp.special.erf(vEscape/v0) - sp.special.erf((vMin - vEarth)/v0)))
          else:
               val.append(0.0)
     
     val = np.array(val)
     K = (v0**3)*(-2.0*pi*(vEscape/v0)*np.exp(-(vEscape/v0)**2) + (pi**1.5)*sp.special.erf(vEscape/v0))
     return (v0**2)*np.pi/(2.0*vEarth*K)*val

""""Calculate F_DM"""
def F_DM(q, FDM_exp):                                                # Unitless
     return (alpha*mElectron/q)**FDM_exp

def TFscreening(q, E, DoScreen):
     if DoScreen==True:
          val = 1.0/(eps0 - 1) + alphaS*((q/qTF)**2) + q**4/(4.*(mElectron**2)*(omegaP**2)) - (E/omegaP)**2
          return 1./(1. + 1./val)
     else:
          return 1.0

"""Define the integrand to perform Simpson's rule"""
def momentum_integrand(q_index, E_index, mX, F_crystal2, FDM_exp, DoScreen):
     qArr, E = dq*q_index + q_shift + dq/2.0, dE*E_index + E_shift + dE/2.0
     return E/(qArr**2)*eta_MB(qArr, E, mX)*(F_DM(qArr, FDM_exp)**2)*F_crystal2[q_index, E_index]*(TFscreening(qArr, E, DoScreen)**2)

"""Integrate the integrand to find dR/d(ln E) for fixed E"""
def d_rate_fixedE(E_index, mX, F_crystal2, FDM_exp, DoScreen):
     prefactor = (rhoX/mX)*(MTarget/MCell)*crosssection*alpha*((mElectron/reducedMass(mX))**2)
     qiArr = np.arange(np.shape(F_crystal2)[0])
     return prefactor*sp.integrate.simps(momentum_integrand(qiArr, E_index, mX, F_crystal2, FDM_exp, DoScreen), x = dq*qiArr + q_shift + dq/2.0)

def Fano_Convert(vals, Earr):
     numE = Earr.shape[0]
     final_numE, final_E, final_vals = (numE - startE)//Ebins, np.append(np.array([0]), np.arange((numE - startE)//Ebins)*dE*Ebins + E_shift + startE*dE), np.array([0.0])
     for i in range(final_numE):
          newval = sp.integrate.simps(vals[Ebins*i + startE: Ebins*(i+1) + startE], x = Earr[Ebins*i : Ebins*(i+1)])
          final_vals = np.append(final_vals, newval)
     return final_vals, final_E

"""Find dR/dE as a function of E"""
def d_rate(mX, F_crystal2, FDM_exp = 0, DoScreen = False):
     numE = np.shape(F_crystal2)[1]
     vals, Earr = np.array([]), np.arange(numE)*dE + E_shift + dE/2.0
     for E_index in np.arange(numE):
          vals = np.append(vals, d_rate_fixedE(E_index, mX, F_crystal2, FDM_exp, DoScreen))
     #intrates, finE = Fano_Convert(vals, Earr)
     return Earr, vals/cm2sec/sec2yr/Earr

"""total rate"""
def rate(mX, F_crystal2, FDM_exp = 0, DoScreen = False):
     numE = np.shape(F_crystal2)[1]
     vals, Earr = np.array([]), np.arange(numE)*dE + E_shift + dE/2.0
     for E_index in np.arange(numE):
          vals = np.append(vals, d_rate_fixedE(E_index, mX, F_crystal2, FDM_exp, DoScreen))
     return sp.integrate.simps(vals/cm2sec/sec2yr/Earr, x = Earr)
