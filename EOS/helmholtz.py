"""
This is a python implementation of the interpolation and thermodynamics
routines used in the helmholtz equation of state.



Go to

http://cococubed.asu.edu/code_pages/eos.shtml

for more details on the helmholtz equation of state by Frank Timmes et. al. Also, see the paper on this code:

F. X. Timmes and F. Douglas Swesty, The Accuracy, Consistency, and Speed of an Electron-Positron Equation of State Based on Table Interpolation of the Helmholtz Free Energy, ApJ 126-2-501 (2000). http://stacks.iop.org/0067-0049/126/i=2/a=501. 


A note on performance:

While it is possible to call this code on individual points of interest, there is
a substantial performance reduction associated with this. Scalar calls take roughly
1.5 milliseconds on a 2015 2.5GHz mobile Core i7. Vectorized calls, on the other hand,
take 4 microseconds each on the same system. As a result, you are strongly encouraged
to assemble all temperature, density, atomic weight, and charge data into vectors
which may be passed to this routine, rather than calling it many times on individual
points.
"""

import sys
import numpy as np
sys.path.append('../')
from sourceClass import *

# Mathematical constants
pi          = np.pi
forpi       = 4*np.pi
eulercon    = 0.577215664901532861 # Euler-Mascheroni
a2rad       = pi/180
rad2a       = 180/pi
third       = 1./3.
forth       = 4./3


# Physical constants
g           = 6.6742867e-8 # Gravitational constant (cm^3/g/s^2)
h           = 6.6260689633e-27 # Planck constant (erg*s)
hbar        = h/(2*pi) # Reduced Planck constant (erg*s)
qe          = 4.8032042712e-10 # Electron charge (esu)
esqu        = qe**2 # Electron charge squared (esu^2)
avo         = 6.0221417930e23 # Avogadro's number (1/mol)
clight      = 2.99792458e10 # Speed of light (cm/s)
light2      = clight**2 # Speed of light squared (cm^2/s^2)
kerg        = 1.380650424e-16 # Boltzmann Constant (erg/K)
amu         = 1.66053878283e-24 # Atomic Mass Unit (g)
mn          = 1.67492721184e-24 # Neutron mass (g)
mp          = 1.67262163783e-24 # Proton mass (g)
me          = 9.1093821545e-28 # Electron mass (g)
rbohr       = hbar*hbar/(me * qe * qe) # Bohr radius (cm)
fine        = qe*qe/(hbar*clight) # Fine structure constant
hion        = 13.605698140 # Rydberg (eV)
ssol        = 5.67051e-5 # Stefan-Boltzmann constant (erg/s/cm^2/K^4)
asol        = 4*ssol/clight # Photon energy density constant (erg/cm^3/K^4)
asoli3      = asol/3. # 1/3 of asol (erg/cm^3/K^4)
weinlam     = h*clight/(kerg*4.965114232) # Wein's Wavelength Law constant (cm*K)
weinfre     = 2.821439372*kerg/h # Wein's Frequency Law constant (1/K/s)
rhonuc      = 2.342e14 # Nuclear density (g/cm^3)
kergavo     = kerg*avo # Avogadro's number*Boltzmann constant (erg/K/mol)
ikavo       = 1./kergavo # Inverse of kergavo (mol*K/erg)
sioncon 	= (2.0 * pi * amu * kerg)/(h**2) # Ion concentration constant (1/cm^2/K)

# Astronomical constants
msol        = 1.9892e33 # Solar mass (g)
rsol        = 6.95997e10 # Solar radius (cm)
lsol        = 3.8268e33 # Solar luminosity (erg/s)
mearth      = 5.9764e27 # Earth mass (g)
rearth      = 6.37e8 # Earth radius (cm)
ly          = 9.460528e17 # Lightyear (cm)
pc          = 3.261633e0 * ly # Parsec (cm)
au          = 1.495978921e13 # Astronomical unit (cm)
secyer      = 3.1558149984e7 # Year (s)

# Coulomb correction constants  
a1 = -0.898004
b1 = 0.96786
c1 = 0.220703
d1 = -0.86097
e1 = 2.5269
a2 = 0.29561
b2 = 1.9885
c2 = 0.288675
		
# Quintic hermite polynomials and derivatives.
def psi0(z):
	return z**3*(z*(-6*z+15)-10)+1
def dpsi0(z):
	return z**2*(z*(-30*z+60)-30)
def ddpsi0(z):
	return z*(z*(-120*z+180)-60)
def psi1(z):
	return z* ( z**2 * ( z * (-3.0*z + 8.0) - 6.0) + 1.0)
def dpsi1(z):
	return z*z * ( z * (-15.0*z + 32.0) - 18.0) +1.0
def ddpsi1(z):
	return z * (z * (-60.0*z + 96.0) -36.0)
def psi2(z):
	return 0.5*z*z*( z* ( z * (-z + 3.0) - 3.0) + 1.0)
def dpsi2(z):
	return 0.5*z*( z*(z*(-5.0*z + 12.0) - 9.0) + 2.0)
def ddpsi2(z):
	return 0.5*(z*( z * (-20.0*z + 36.0) - 18.0) + 2.0)

def fiLookup(tables,i,j,names):
	# i and j must be one-dimensional numpy arrays of integers.
	fi = np.zeros((len(i),4*len(names)))
	for k in range(len(names)):
		fi[:,4*k] = tables[names[k]][i,j]
		fi[:,4*k+1] = tables[names[k]][i+1,j]
		fi[:,4*k+2] = tables[names[k]][i,j+1]
		fi[:,4*k+3] = tables[names[k]][i+1,j+1]
	return np.transpose(fi)

# Biquintic hermite polynomial (for interpolation)
def h5(fi,w0t,w1t,w2t,w0mt,w1mt,w2mt,w0d,w1d,w2d,w0md,w1md,w2md):
	return fi[0]  *w0d*w0t   + fi[1]  *w0md*w0t \
		   + fi[2]  *w0d*w0mt  + fi[3]  *w0md*w0mt \
		   + fi[4]  *w0d*w1t   + fi[5]  *w0md*w1t \
		   + fi[6]  *w0d*w1mt  + fi[7]  *w0md*w1mt \
		   + fi[8]  *w0d*w2t   + fi[9] *w0md*w2t \
		   + fi[10] *w0d*w2mt  + fi[11] *w0md*w2mt \
		   + fi[12] *w1d*w0t   + fi[13] *w1md*w0t \
		   + fi[14] *w1d*w0mt  + fi[15] *w1md*w0mt \
		   + fi[16] *w2d*w0t   + fi[17] *w2md*w0t \
		   + fi[18] *w2d*w0mt  + fi[19] *w2md*w0mt \
		   + fi[20] *w1d*w1t   + fi[21] *w1md*w1t \
		   + fi[22] *w1d*w1mt  + fi[23] *w1md*w1mt \
		   + fi[24] *w2d*w1t   + fi[25] *w2md*w1t \
		   + fi[26] *w2d*w1mt  + fi[27] *w2md*w1mt \
		   + fi[28] *w1d*w2t   + fi[29] *w1md*w2t \
		   + fi[30] *w1d*w2mt  + fi[31] *w1md*w2mt \
		   + fi[32] *w2d*w2t   + fi[33] *w2md*w2t \
		   + fi[34] *w2d*w2mt  + fi[35] *w2md*w2mt

# Bicubic hermite polynomial (for interpolation)
def h3(fi,w0t,w1t,w0mt,w1mt,w0d,w1d,w0md,w1md):
	return fi[0]  *w0d*w0t   +  fi[1]  *w0md*w0t \
		   + fi[2]  *w0d*w0mt  +  fi[3]  *w0md*w0mt \
		   + fi[4]  *w0d*w1t   +  fi[5]  *w0md*w1t \
		   + fi[6]  *w0d*w1mt  +  fi[7]  *w0md*w1mt \
		   + fi[8]  *w1d*w0t   +  fi[9] *w1md*w0t \
		   + fi[10] *w1d*w0mt  +  fi[11] *w1md*w0mt \
		   + fi[12] *w1d*w1t   +  fi[13] *w1md*w1t \
		   + fi[14] *w1d*w1mt  +  fi[15] *w1md*w1mt

# Cubic hermite polynomial
def xpsi0(z):
	return z*z*(2*z-3)+1
def xdpsi0(z):
	return z*(6*z-6)
def xpsi1(z):
	return z*(z*(z-2)+1)
def xdpsi1(z):
	return z*(3*z-4)+1

# Helper for computing abar, zbar in the simple case of H, He, and C:
def azHelper(ionmax=3,xmass=np.array([0.75,0.23,0.02]),\
					aion=np.array([1.0,4.0,12.0]),zion=np.array([1.0,2.0,6.0])):
		"""
		# Compute the average atomic weight and charge

		Arguments:
		ionmax  - Number of isotopes in the network (default is 3). Should be an integer.
		xmass   - Mass fraction of each isotope (default is np.array([0.75,0.23,0.02])). Should be a float numpy array.
		aion    - Number of nucleons in each isotope (default is np.array([1.0,4.0,12.0])). Should be a float numpy array.
		zion    - Number of protons in each isotope (default is np.array([1.0,2.0,6.0])). Should be a float numpy array.
		Default arguments for the above are given assuming a composition of hydrogen, helium, and carbon.
		"""
		abar = 1./np.sum(xmass/aion)
		zbar = abar*sum(xmass*zion/aion)
		return abar, zbar


def helmSource(fname='SourceTables/helmholtz/helm_table.dat',logTmin=3.0,logTmax=13.0,\
				logRhoMin=-12.0,logRhoMax=15.0,tRes=101,rhoRes=271):
	"""
	This method constructs a source object for the Helmholtz tables.

	Arguments:

	fname   - Filename of table. Should be a string.

	logTmin - Log10 of minimum temperature in the tables (K). Should be a float.
	logTmax - Log10 of maximum temperature in the tables (K). Should be a float.
	tRes    - Number of grid points in temperature in the tables. Should be an integer.
	logRhoMin   -   Log10 of minimum density in the tables (g/cm^3). Should be a float.
	logRhoMax   -   Log10 of maximum density in the tables (g/cm^3). Should be a float.
	rhoRes  - Number of grid points in density in the tables. Should be an integer.

	"""

	# Table limits and declaration
	logTRan = np.linspace(logTmin,logTmax,num=tRes,endpoint=True)
	logRhoRan = np.linspace(logRhoMin,logRhoMax,num=rhoRes,endpoint=True)
	logTstep = logTRan[1] - logTRan[0]
	logRhoStep = logRhoRan[1] - logRhoRan[0]
	tRan = 10**logTRan
	rhoRan = 10**logRhoRan


	# We store the tables as a dictionary, with named 2D tables corresponding to the stored variables
	names = ['f','fd','ft','fdd','ftt','fdt','fddt','fdtt','fddtt','dpdf','dpdfd'\
			,'dpdft','dpdfdt','ef','efd','eft','efdt','xf','xfd','xft','xfdt']
	tables = {n:np.zeros((rhoRan.shape[0],tRan.shape[0])) for n in names}

	# Open Helmholtz table for reading
	fi = open(fname)

	# Read the Helmholtz free energy and its derivatives
	for j in range(len(tRan)):
		for i in range(len(rhoRan)):
			s = fi.readline().rstrip('\n')[2:].split(' ')
			s = [float(a) for a in s if a!='' and a!='\n']
			for k in range(9):
				tables[names[k]][i,j] = s[k]

	# Read the pressure derivative with density
	for j in range(len(tRan)):
		for i in range(len(rhoRan)):
			s = fi.readline().rstrip('\n')[2:].split(' ')
			s = [a for a in s if a!='' and a!='\n']
			for k in range(4):
				tables[names[9+k]][i,j] = s[k]

	# Read the electron chemical potentials
	for j in range(len(tRan)):
		for i in range(len(rhoRan)):
			s = fi.readline().rstrip('\n')[2:].split(' ')
			s = [a for a in s if a!='' and a!='\n']
			for k in range(4):
				tables[names[13+k]][i,j] = s[k]

	# Read the nuclear densities
	for j in range(len(tRan)):
		for i in range(len(rhoRan)):
			s = fi.readline().rstrip('\n')[2:].split(' ')
			s = [a for a in s if a!='' and a!='\n']
			for k in range(4):
				tables[names[17+k]][i,j] = s[k]

	fi.close()

	# Construct the temperature and density deltas and their inverses
	dT = np.diff(tRan)
	dT2 = dT**2
	dT3 = dT**3
	dTinverse = 1./dT
	dT2inverse = 1./dT2
	dT3inverse = 1./dT3
	dRho = np.diff(rhoRan)
	dRho2 = dRho**2
	dRho3 = dRho**3
	dRhoInverse = 1./dRho
	dRho2Inverse = 1./dRho2
	dRho3Inverse = 1./dRho3

	# Return names - These are left unchanged from the original fortran code to make it easier to read this code.
	# The filtered method takes (and defines) a specific subset of these needed for most stellar codes.
	outNames = ['P','dP/dT|Rho','dP/dRho|T','dP/dA','dP/dZ','E','dE/dT|Rho','dE/dRho|T','dE/dA','dE/dZ','S','dS/dT|Rho',\
					 'dS/dRho|T','dS/dA','dS/dZ','Pgas','dPgas/dT|Rho','dPgas/dRho|T','dPgas/dA','dPgas/dZ','Egas','dEgas/dT|Rho',\
					 'dEgas/dRho|T','dEgas/dA','dEgas/dZ','Prad','dPrad/dT|Rho','dPrad/dRho|T','dPrad/dA','dPrad/dZ','Erad',\
					 'dErad/dT|Rho','dErad/dRho|T','dErad/dA','dErad/dZ','Srad','dSrad/dT|Rho','dSrad/dRho|T','dSrad/dA','dSrad/dZ',\
					 'Pion','dPion/dT|Rho','dPion/dRho|T','dPion/dA','dPion/dZ','Eion','dEion/dT|Rho','dEion/dRho|T','dEion/dA',\
					 'dEion/dZ','Sion','dSion/dT|Rho','dSion/dRho|T','dSion/dA','dSion/dZ','Xni','Pele','Ppos','dPep/dT|Rho','dPep/dRho|T',\
					 'dPep/dA','dPep/dZ','Eele','Epos','dEep/dT|Rho','dEep/dRho|T','dEep/dA','dEep/dZ','Sele','Spos','dSep/dT|Rho','dSep/dT|Rho',\
					 'dSep/dA','dSep/dZ','Xnem','Xnefer','dXne/dT|Rho','dXne/dRho|T','dXne/dA','dXne/dZ','Xnp','Zbar','Etaele','dEta/dT|Rho',\
					 'dEta/dRho|T','dEta/dA','dEta/dZ','Etapos','Pcoul','dPcoul/dT|Rho','dPcoul/dRho|T','dPcoul/dA','dPcoul/dZ','Ecoul',\
					 'dEcoul/dT|Rho','dEcoul/dRho|T','dEcoul/dA','dEcoul/dZ','Scoul','dScoul/dT|Rho','dScoul/dRho|T','dScoul/dA','dScoul/dZ',\
					 'PlasGamma','dSe','dPe','dSp','CVgas','CPgas','gamma1gas','gamma2gas','gamma3gas','dLogT/dlogPgas|S','vsGas','CV','CP','gamma1',\
					 'gamma2','gamma3','dLogT/dlogP|S','vs']

	def contains(points):

		# Sanitize inputs so that only numpy arrays come in

		temp = points[:,0]
		den = points[:,1]
		abar = points[:,2]
		zbar = points[:,3]

		return 1-1.0*((temp<10**logTmin) | (temp>10**logTmax) | \
					(den<10**logRhoMin) | (den>10**logRhoMax) | (abar<1) | (zbar<1))

	def smoothMask(points):

		ret = contains(points)

		# Sanitize inputs so that only numpy arrays come in

		temp = points[:,0]
		den = points[:,1]
		abar = points[:,2]
		zbar = points[:,3]

		x = (np.log10(temp)-logTmin)/(logTmax-logTmin)
		ret *= np.maximum(0,1-2./(1+np.exp(30*x))-2./(1+np.exp(30*(1-x)))) # 30 was picked so that the transition happens over ~10% of the range
		x = (np.log10(den)-logRhoMin)/(logRhoMax-logRhoMin)
		ret *= np.maximum(0,1-2./(1+np.exp(30*x))-2./(1+np.exp(30*(1-x)))) # 30 was picked so that the transition happens over ~10% of the range

		# Arbitrary abar>0, zbar>1 are allowed.
		x = abar-1+1e-10
		ret *= np.maximum(0,1-2./(1+np.exp(30*x))) # 30 was picked so that the transition happens over ~10% of the range
		x = zbar-1+1e-10 # The 1e-10 we add to give you some leeway if you really do want a pure hydrogen gas
		ret *= np.maximum(0,1-2./(1+np.exp(30*x))) # 30 was picked so that the transition happens over ~10% of the range

		return ret

	def data(points):
		"""
		Computes the equation of state and returns the various thermodynamic quantities.

		Arguments:
		points - 2D Numpy array of shape (N,4). The four columns are:
			temp - Temperature (K).
			den  - Density (g/cm^3).
			abar - Average atomic weight of nuclei (in proton masses).
			zion - Average Z among the isotopes present.

		The output of this method is a dictionary, the values of which are one-dimensional numpy arrays.
		The keys are in outNames, and the corresponding definitions are given in the variable declarations file.

		All arguments should either be floats or 1-dimensional numpy arrays. Mixed inputs
		between floats and arrays are allowed so long as each array present has the same length.
		The return value will either be a 2-dimensional vector.
		"""

		# Sanitize inputs so that only numpy arrays come in

		temp = points[:,0]
		den = points[:,1]
		abar = points[:,2]
		zbar = points[:,3]

		maxLen = max(len(temp),len(den),len(abar),len(zbar))
		if maxLen > 1:
			if len(temp) == 1:
				temp = temp[0]*np.ones(maxLen)
			if len(den) == 1:
				den = den[0]*np.ones(maxLen)
			if len(abar) == 1:
				abar = abar[0]*np.ones(maxLen)
			if len(zbar) == 1:
				zbar = zbar[0]*np.ones(maxLen)	

		# Bomb-proof the input... we'll set all outputs corresponding to bad inputs to NaN
		# at the end.
		nanLocs = np.where((temp<10**logTmin) | (temp>10**logTmax) | \
					(den<10**logRhoMin) | (den>10**logRhoMax))
		temp[temp<10**logTmin] = 10**logTmin
		temp[temp>10**logTmax] = 10**logTmax
		den[den<10**logRhoMin] = 10**logRhoMin
		den[den>10**logRhoMax] = 10**logRhoMax

		# Initialize
		ytot1 = 1./abar
		ye = np.maximum(1e-16*np.ones(temp.shape),ytot1*zbar)
		deni = 1./den
		tempi = 1./temp
		kt = kerg*temp
		ktinv = 1./kt

		# Radiation
		prad = asoli3*temp**4
		dpraddd = np.zeros(temp.shape)
		dpraddt = 4.0 * prad*tempi
		dpradda = np.zeros(temp.shape)
		dpraddz = np.zeros(temp.shape)

		erad    = 3.0 * prad*deni
		deraddd = -erad*deni
		deraddt = 3.0 * dpraddt*deni
		deradda = np.zeros(temp.shape)
		deraddz = np.zeros(temp.shape)

		srad    = (prad*deni + erad)*tempi
		dsraddd = (dpraddd*deni - prad*deni*deni + deraddd)*tempi
		dsraddt = (dpraddt*deni + deraddt - srad)*tempi
		dsradda = np.zeros(temp.shape)
		dsraddz = np.zeros(temp.shape)

		# Ions
		xni     = avo * ytot1 * den
		dxnidd  = avo * ytot1
		dxnida  = -xni * ytot1

		pion    = xni * kt
		dpiondd = dxnidd * kt
		dpiondt = xni * kerg
		dpionda = dxnida * kt
		dpiondz = np.zeros(temp.shape)

		eion    = 1.5 * pion*deni
		deiondd = (1.5 * dpiondd - eion)*deni
		deiondt = 1.5 * dpiondt*deni
		deionda = 1.5 * dpionda*deni
		deiondz = np.zeros(temp.shape)

		# Sackur-Tetrode equation for the ion entropy of a single ideal gas characterized by abar
		x       = abar*abar*np.sqrt(abar) * deni/avo
		s       = sioncon * temp
		z       = x * s * np.sqrt(s)
		y       = np.log(z)

		sion    = (pion*deni + eion)*tempi + kergavo * ytot1 * y
		dsiondd = (dpiondd*deni - pion*deni*deni + deiondd)*tempi \
				   - kergavo * deni * ytot1
		dsiondt = (dpiondt*deni + deiondt)*tempi - \
				  (pion*deni + eion) * tempi*tempi \
				  + 1.5 * kergavo * tempi*ytot1
		x       = avo*kerg/abar
		dsionda = (dpionda*deni + deionda)*tempi \
				  + kergavo*ytot1*ytot1* (2.5 - y)
		dsiondz = np.zeros(temp.shape)

		# Assume complete ionization
		xnem = xni*zbar

		# Enter the table with ye*den
		din = ye*den

		# Locate this temperature and density
		j = np.floor(((np.log10(temp)-logTmin)/logTstep)).astype(int)
		i = np.floor(((np.log10(din)-logRhoMin)/logRhoStep)).astype(int)
		i[i<0] = 0
		i[i>=rhoRes] = rhoRes - 1
		j[j<0] = 0
		j[j>=tRes] = tRes - 1

		# Access the relevant portions of the table
		fi = fiLookup(tables,i,j,['f','ft','ftt','fd','fdd','fdt','fddt','fdtt','fddtt'])

		# Various differences
		xt  = np.maximum((temp - tRan[j])*dTinverse[j], np.zeros(temp.shape))
		xd  = np.maximum((din - rhoRan[i])*dRhoInverse[i], np.zeros(din.shape))
		mxt = 1.0 - xt
		mxd = 1.0 - xd

		# The six density and six temperature basis functions
		si0t =   psi0(xt)
		si1t =   psi1(xt)*dT[j]
		si2t =   psi2(xt)*dT2[j]

		si0mt =  psi0(mxt)
		si1mt = -psi1(mxt)*dT[j]
		si2mt =  psi2(mxt)*dT2[j]

		si0d =   psi0(xd)
		si1d =   psi1(xd)*dRho[i]
		si2d =   psi2(xd)*dRho2[i]

		si0md =  psi0(mxd)
		si1md = -psi1(mxd)*dRho[i]
		si2md =  psi2(mxd)*dRho2[i]

		# Derivatives of the weight functions
		dsi0t =   dpsi0(xt)*dTinverse[j]
		dsi1t =   dpsi1(xt)
		dsi2t =   dpsi2(xt)*dT[j]

		dsi0mt = -dpsi0(mxt)*dTinverse[j]
		dsi1mt =  dpsi1(mxt)
		dsi2mt = -dpsi2(mxt)*dT[j]

		dsi0d =   dpsi0(xd)*dRhoInverse[i]
		dsi1d =   dpsi1(xd)
		dsi2d =   dpsi2(xd)*dRho[i]

		dsi0md = -dpsi0(mxd)*dRhoInverse[i]
		dsi1md =  dpsi1(mxd)
		dsi2md = -dpsi2(mxd)*dRho[i]

		# Second derivatives of the weight functions
		ddsi0t =   ddpsi0(xt)*dT2inverse[j]
		ddsi1t =   ddpsi1(xt)*dTinverse[j]
		ddsi2t =   ddpsi2(xt)

		ddsi0mt =  ddpsi0(mxt)*dT2inverse[j]
		ddsi1mt = -ddpsi1(mxt)*dTinverse[j]
		ddsi2mt =  ddpsi2(mxt)

		# The free energy
		free = h5(fi, \
				si0t,   si1t,   si2t,   si0mt,   si1mt,   si2mt, \
				si0d,   si1d,   si2d,   si0md,   si1md,   si2md)

		# Derivative with respect to density
		df_d  = h5(fi, \
				si0t,   si1t,   si2t,   si0mt,   si1mt,   si2mt, \
				dsi0d,  dsi1d,  dsi2d,  dsi0md,  dsi1md,  dsi2md)

		# Derivative with respect to temperature
		df_t = h5(fi, \
				dsi0t,  dsi1t,  dsi2t,  dsi0mt,  dsi1mt,  dsi2mt, \
				si0d,   si1d,   si2d,   si0md,   si1md,   si2md)

		# Derivative with respect to temperature (twice)
		df_tt = h5(fi, \
			  ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt, \
				si0d,   si1d,   si2d,   si0md,   si1md,   si2md)

		# Derivative with respect to temperature and density
		df_dt = h5(fi, \
				dsi0t,  dsi1t,  dsi2t,  dsi0mt,  dsi1mt,  dsi2mt, \
				dsi0d,  dsi1d,  dsi2d,  dsi0md,  dsi1md,  dsi2md)

		# Now get the pressure derivative with density, chemical potential, and electron positron number densities
		# Get the interpolation weight functions
		si0t   =  xpsi0(xt)
		si1t   =  xpsi1(xt)*dT[j]

		si0mt  =  xpsi0(mxt)
		si1mt  =  -xpsi1(mxt)*dT[j]

		si0d   =  xpsi0(xd)
		si1d   =  xpsi1(xd)*dRho[i]

		si0md  =  xpsi0(mxd)
		si1md  =  -xpsi1(mxd)*dRho[i]

		# Derivatives of weight functions
		dsi0t  = xdpsi0(xt)*dTinverse[j]
		dsi1t  = xdpsi1(xt)

		dsi0mt = -xdpsi0(mxt)*dTinverse[j]
		dsi1mt = xdpsi1(mxt)

		dsi0d  = xdpsi0(xd)*dRhoInverse[i]
		dsi1d  = xdpsi1(xd)

		dsi0md = -xdpsi0(mxd)*dRhoInverse[i]
		dsi1md = xdpsi1(mxd)

		# Access the relevant portions of the table
		fi = fiLookup(tables,i,j,['dpdf','dpdft','dpdfd','dpdfdt'])

		# Pressure derivative with density
		dpepdd  = h3(fi, \
					   si0t,   si1t,   si0mt,   si1mt, \
					   si0d,   si1d,   si0md,   si1md)
		dpepdd  = np.maximum(ye * dpepdd,1.0e-30)

		# Access the relevant portions of the table
		fi = fiLookup(tables,i,j,['ef','eft','efd','efdt'])

		# Electron chemcial potential etaele
		etaele  = h3(fi, \
					 si0t,   si1t,   si0mt,   si1mt, \
					 si0d,   si1d,   si0md,   si1md)

		# Derivative with respect to density
		x       = h3(fi, \
					 si0t,   si1t,   si0mt,   si1mt, \
					dsi0d,  dsi1d,  dsi0md,  dsi1md)
		detadd  = ye * x

		# Derivative with respect to temperature
		detadt  = h3(fi, \
					dsi0t,  dsi1t,  dsi0mt,  dsi1mt, \
					 si0d,   si1d,   si0md,   si1md)

		# Derivative with respect to abar and zbar
		detada = -x * din * ytot1
		detadz =  x * den * ytot1

	   # Access the relevant portions of the table
		fi = fiLookup(tables,i,j,['xf','xft','xfd','xfdt'])

		# Electron and positron number densities
		xnefer   = h3(fi, \
					 si0t,   si1t,   si0mt,   si1mt, \
					 si0d,   si1d,   si0md,   si1md)

		# Derivative with respect to density
		x        = h3(fi, \
					 si0t,   si1t,   si0mt,   si1mt, \
					dsi0d,  dsi1d,  dsi0md,  dsi1md)
		x = np.maximum(x,1.0e-30)
		dxnedd   = ye * x

		# Derivative with respect to temperature
		dxnedt   = h3(fi, \
					dsi0t,  dsi1t,  dsi0mt,  dsi1mt, \
					 si0d,   si1d,   si0md,   si1md)

		# Derivative with respect to abar and zbar
		dxneda = -x * din * ytot1
		dxnedz =  x  * den * ytot1

		# Desired electron-positron thermodynamic quantities
		# dpepdd at high temperatures and low densities is below the
		# floating point limit of the subtraction of two large terms.
		# since dpresdd doesn't enter the maxwell relations at all, use the
		# bicubic interpolation done above instead of the formally correct expression
		x       = din * din
		pele    = x * df_d
		dpepdt  = x * df_dt
		s       = dpepdd/ye - 2.0 * din * df_d
		dpepda  = -ytot1 * (2.0 * pele + s * din)
		dpepdz  = den*ytot1*(2.0 * din * df_d  +  s)

		x       = ye * ye
		sele    = -df_t * ye
		dsepdt  = -df_tt * ye
		dsepdd  = -df_dt * x
		dsepda  = ytot1 * (ye * df_dt * din - sele)
		dsepdz  = -ytot1 * (ye * df_dt * den  + df_t)

		eele    = ye*free + temp * sele
		deepdt  = temp * dsepdt
		deepdd  = x * df_d + temp * dsepdd
		deepda  = -ye * ytot1 * (free +  df_d * din) + temp * dsepda
		deepdz  = ytot1* (free + ye * df_d * den) + temp * dsepdz

		# Coulomb section
		# uniform background corrections only
		# from yakovlev & shalybkov 1989
		# lami is the average ion seperation
		# plasg is the plasma coupling parameter

		z        = forth * pi
		s        = z * xni
		dsdd     = z * dxnidd
		dsda     = z * dxnida

		lami     = 1.0/s**third
		inv_lami = 1.0/lami
		z        = -third * lami
		lamidd   = z * dsdd/s
		lamida   = z * dsda/s

		plasg    = zbar*zbar*esqu*ktinv*inv_lami
		z        = -plasg * inv_lami
		plasgdd  = z * lamidd
		plasgda  = z * lamida
		plasgdt  = -plasg*ktinv * kerg
		plasgdz  = 2.0 * plasg/zbar

		# yakovlev & shalybkov 1989 equations 82, 85, 86, 87
		pcoul = np.zeros(temp.shape)
		ecoul = np.zeros(temp.shape)
		scoul = np.zeros(temp.shape)
		decouldd = np.zeros(temp.shape)
		decouldt = np.zeros(temp.shape)
		decoulda = np.zeros(temp.shape)
		decouldz = np.zeros(temp.shape)
		dpcouldd = np.zeros(temp.shape)
		dpcouldt = np.zeros(temp.shape)
		dpcoulda = np.zeros(temp.shape)
		dpcouldz = np.zeros(temp.shape)
		dscouldd = np.zeros(temp.shape)
		dscouldt = np.zeros(temp.shape)
		dscoulda = np.zeros(temp.shape)
		dscouldz = np.zeros(temp.shape)

		slicePlas = np.where(plasg > 1.)
		if len(slicePlas[0]) > 0:	
			x[slicePlas] = (plasg**0.25)[slicePlas]
			y[slicePlas] = (avo * ytot1 * kerg)[slicePlas]
			ecoul[slicePlas] = (y * temp * (a1*plasg + b1*x + c1/x + d1))[slicePlas]
			pcoul[slicePlas] = (third * den * ecoul)[slicePlas]
			scoul[slicePlas] = ( -y * (3.0*b1*x - 5.0*c1/x + d1 * (np.log(plasg) - 1.0) - e1))[slicePlas]

			y[slicePlas] = (avo*ytot1*kt*(a1 + 0.25/plasg*(b1*x - c1/x)))[slicePlas]
			decouldd[slicePlas] = (y * plasgdd)[slicePlas]
			decouldt[slicePlas] = (y * plasgdt + ecoul/temp)[slicePlas]
			decoulda[slicePlas] = (y * plasgda - ecoul/abar)[slicePlas]
			decouldz[slicePlas] = (y * plasgdz)[slicePlas]

			y[slicePlas]        = (third * den)[slicePlas]
			dpcouldd[slicePlas] = (third * ecoul + y*decouldd)[slicePlas]
			dpcouldt[slicePlas] = (y * decouldt)[slicePlas]
			dpcoulda[slicePlas] = (y * decoulda)[slicePlas]
			dpcouldz[slicePlas] = (y * decouldz)[slicePlas]

			y[slicePlas]        = (-avo*kerg/(abar*plasg)*(0.75*b1*x+1.25*c1/x+d1))[slicePlas]
			dscouldd[slicePlas] = (y * plasgdd)[slicePlas]
			dscouldt[slicePlas] = (y * plasgdt)[slicePlas]
			dscoulda[slicePlas] = (y * plasgda - scoul/abar)[slicePlas]
			dscouldz[slicePlas] = (y * plasgdz)[slicePlas]

		slicePlas = np.where(plasg < 1.)
		if len(slicePlas[0]) > 0:	
			x[slicePlas]        = (plasg*np.sqrt(plasg))[slicePlas]
			y[slicePlas]        = (plasg**b2)[slicePlas]
			z[slicePlas]        = (c2 * x - third * a2 * y)[slicePlas]
			pcoul[slicePlas]    = (-pion * z)[slicePlas]
			ecoul[slicePlas]    = (3.0 * pcoul/den)[slicePlas]
			scoul[slicePlas]    = (-avo/abar*kerg*(c2*x -a2*(b2-1.0)/b2*y))[slicePlas]

			s[slicePlas]        = (1.5*c2*x/plasg - third*a2*b2*y/plasg)[slicePlas]
			dpcouldd[slicePlas] = (-dpiondd*z - pion*s*plasgdd)[slicePlas]
			dpcouldt[slicePlas] = (-dpiondt*z - pion*s*plasgdt)[slicePlas]
			dpcoulda[slicePlas] = (-dpionda*z - pion*s*plasgda)[slicePlas]
			dpcouldz[slicePlas] = (-dpiondz*z - pion*s*plasgdz)[slicePlas]

			s[slicePlas]        = (3.0/den)[slicePlas]
			decouldd[slicePlas] = (s * dpcouldd - ecoul/den)[slicePlas]
			decouldt[slicePlas] = (s * dpcouldt)[slicePlas]
			decoulda[slicePlas] = (s * dpcoulda)[slicePlas]
			decouldz[slicePlas] = (s * dpcouldz)[slicePlas]

			s[slicePlas]        = (-avo*kerg/(abar*plasg)*(1.5*c2*x-a2*(b2-1.0)*y))[slicePlas]
			dscouldd[slicePlas] = (s * plasgdd)[slicePlas]
			dscouldt[slicePlas] = (s * plasgdt)[slicePlas]
			dscoulda[slicePlas] = (s * plasgda - scoul/abar)[slicePlas]
			dscouldz[slicePlas] = (s * plasgdz)[slicePlas]

		# Bomb-proof
		x   = prad + pion + pele + pcoul
		y   = erad + eion + eele + ecoul
		z   = srad + sion + sele + scoul

		sliceB = np.where((x<0) | (y<0) | (z<0))
		pcoul[sliceB]    = 0.0
		dpcouldd[sliceB] = 0.0
		dpcouldt[sliceB] = 0.0
		dpcoulda[sliceB] = 0.0
		dpcouldz[sliceB] = 0.0
		ecoul[sliceB]    = 0.0
		decouldd[sliceB] = 0.0
		decouldt[sliceB] = 0.0
		decoulda[sliceB] = 0.0
		decouldz[sliceB] = 0.0
		scoul[sliceB]    = 0.0
		dscouldd[sliceB] = 0.0
		dscouldt[sliceB] = 0.0
		dscoulda[sliceB] = 0.0
		dscouldz[sliceB] = 0.0

		# Sum all the gas components
		pgas    = pion + pele + pcoul
		egas    = eion + eele + ecoul
		sgas    = sion + sele + scoul

		dpgasdd = dpiondd + dpepdd + dpcouldd
		dpgasdt = dpiondt + dpepdt + dpcouldt
		dpgasda = dpionda + dpepda + dpcoulda
		dpgasdz = dpiondz + dpepdz + dpcouldz

		degasdd = deiondd + deepdd + decouldd
		degasdt = deiondt + deepdt + decouldt

		degasda = deionda + deepda + decoulda
		degasdz = deiondz + deepdz + decouldz

		dsgasdd = dsiondd + dsepdd + dscouldd
		dsgasdt = dsiondt + dsepdt + dscouldt
		dsgasda = dsionda + dsepda + dscoulda
		dsgasdz = dsiondz + dsepdz + dscouldz

		# Add in radiation to get the total
		pres    = prad + pgas
		ener    = erad + egas
		entr    = srad + sgas

		dpresdd = dpraddd + dpgasdd
		dpresdt = dpraddt + dpgasdt
		dpresda = dpradda + dpgasda
		dpresdz = dpraddz + dpgasdz

		denerdd = deraddd + degasdd
		denerdt = deraddt + degasdt
		denerda = deradda + degasda
		denerdz = deraddz + degasdz

		dentrdd = dsraddd + dsgasdd
		dentrdt = dsraddt + dsgasdt
		dentrda = dsradda + dsgasda
		dentrdz = dsraddz + dsgasdz

		# for the gas
		# the temperature and density exponents (c&g 9.81 9.82)
		# the specific heat at constant volume (c&g 9.92)
		# the third adiabatic exponent (c&g 9.93)
		# the first adiabatic exponent (c&g 9.97)
		# the second adiabatic exponent (c&g 9.105)
		# the specific heat at constant pressure (c&g 9.98)
		# and relativistic formula for the sound speed (c&g 14.29)
		zz        = pgas*deni
		zzi       = den/pgas
		chit_gas  = temp/pgas * dpgasdt
		chid_gas  = dpgasdd*zzi
		cv_gas    = degasdt
		x         = zz * chit_gas/(temp * cv_gas)
		gam3_gas  = x + 1.0
		gam1_gas  = chit_gas*x + chid_gas
		nabad_gas = x/gam1_gas
		gam2_gas  = 1.0/(1.0 - nabad_gas)
		cp_gas    = cv_gas * gam1_gas/chid_gas
		z         = 1.0 + (egas + light2)*zzi
		sound_gas = clight * np.sqrt(gam1_gas/z)

		# For the totals
		zz    = pres*deni
		zzi   = den/pres
		chit  = temp/pres * dpresdt
		chid  = dpresdd*zzi
		cv    = denerdt
		x     = zz * chit/(temp * cv)
		gam3  = x + 1.0
		gam1  = chit*x + chid
		nabad = x/gam1
		gam2  = 1.0/(1.0 - nabad)
		cp    = cv * gam1/chid
		z     = 1.0 + (ener + light2)*zzi
		sound = clight * np.sqrt(gam1/z)

		# maxwell relations; each is zero if the consistency is perfect
		x   = den * den

		dse = temp*dentrdt/denerdt - 1.0

		dpe = (denerdd*x + temp*dpresdt)/pres - 1.0

		dsp = -dentrdd*x/dpresdt - 1.0

		ret = np.transpose(np.array([pres,dpresdt,dpresdd,dpresda,dpresdz,ener,denerdt,\
						denerdd,denerda,denerdz,entr,dentrdt,dentrdd,dentrda,\
						dentrdz,pgas,dpgasdt,dpgasdd,dpgasda,dpgasdz,egas,degasdt,\
						degasdd,degasda,degasdz,prad,dpraddt,dpraddd,dpradda,dpraddz,\
						erad,deraddt,deraddd,deradda,deraddz,srad,dsraddt,dsraddd,\
						dsradda,dsraddz,pion,dpiondt,dpiondd,dpionda,dpiondz,eion,\
						deiondt,deiondd,deionda,deiondz,sion,dsiondt,dsiondd,dsionda,\
						dsiondz,xni,pele,np.zeros(pres.shape),dpepdt,dpepdd,dpepda,\
						dpepdz,eele,np.zeros(pres.shape),deepdt,deepdd,deepda,deepdz,\
						sele,np.zeros(pres.shape),dsepdt,dsepdd,dsepda,dsepdz,xnem,xnefer,\
						dxnedt,dxnedd,dxneda,dxnedz,np.zeros(pres.shape),zbar,etaele,\
						detadt,detadd,detada,detadz,np.zeros(pres.shape),pcoul,dpcouldt,\
						dpcouldd,dpcoulda,dpcouldz,ecoul,decouldt,decouldd,decoulda,\
						decouldz,scoul,dscouldt,dscouldd,dscoulda,dscouldz,plasg,dse,dpe,\
						dsp,cv_gas,cp_gas,gam1_gas,gam2_gas,gam3_gas,nabad_gas,sound_gas,\
						cv,cp,gam1,gam2,gam3,nabad,sound]))

		ret[nanLocs] = np.nan # Set outputs corresponding to bad input locations to NaN

		return ret

	return source(['T','Rho','Abar','Zbar'],outNames,contains,smoothMask,data)
# (fname='SourceTables/helmholtz/helm_table.dat',logTmin=3.0,logTmax=13.0,\
#				logRhoMin=-12.0,logRhoMax=15.0,tRes=101,rhoRes=271):
h = helmSource()
inVar = 'Rho'
outVar = 'P'
ranges = [10**np.linspace(3.,13.,num=30),np.linspace(1.,10.,num=10),np.linspace(1.,10.,num=10)]
inMin = 1e-12
inMax = 1e15
outRange = 10**np.linspace(-10.,10.,num=100)
s = inversion(h,inVar,outVar,ranges,inMin,inMax,outRange)