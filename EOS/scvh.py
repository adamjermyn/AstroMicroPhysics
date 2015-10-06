"""
This is a python implementation of the interpolation and thermodynamics routines used in the SCVH
equation of state by Saumon, D., Chabrier, G., and Van Horn, H.M.. The tables include the full
documentation of the tables.

See the following references for more information on the SCVH EOS:

Chabrier, G. 1990, J. Phys. (Paris) 51, 1607.
Saumon, D. and Chabrier, G. 1991, Phys. Rev. A 44, 5122.
Saumon, D. and Chabrier, G. 1992, Phys. Rev. A 46, 2084.
Saumon, D., Chabrier, G., and Van Horn, H.M. 1994,  Ap. J. Supp., in press.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline

def readTables(fname):
	logTvals = [] # Temperature gridpoints
	logPvals = [] # Pressure gridpoints
	hTable = [] # Hydrogen table
	maxLen = 0
	
	fi = open(fname)

	for line in fi:
		s = line.rstrip('\n').rstrip('\r').split(' ')
		s = [i for i in s if i != '']
		if len(s) == 2: # Means the start of a new temperature
			logTvals.append(float(s[0]))
			hTable.append([])
		else:
			hTable[-1].append(map(float,s))
		if len(hTable[-1]) > maxLen:
			maxLen = len(hTable[-1])
			logPvals = [hTable[-1][i][0] for i in range(len(hTable[-1]))]

	# The tables always start at the same pressure, and step in the same increments,
	# but the endpoint is not uniform. For efficiency we are going to pad the table such
	# that it is rectangular.
	for i in range(len(hTable)):
		for j in range(maxLen-len(hTable[i])):
			hTable[i].append([np.nan for k in range(len(hTable[0][0]))])

	# Turn it all into NumPy arrays:
	logTvals = np.array(logTvals)
	logPvals = np.array(logPvals)
	hTable = np.array(hTable)
	hTable = hTable[...,1:] # Remove pressure values from table entries

	return hTable,logTvals,logPvals

def sanitizeInput(temp,pressure):
	"""
	This function takes mixed inputs, each of which is either a float or a one-dimensional
	float numpy array, and returns both inputs as one-dimensional float numpy arrays.

	Arguments:
		temp 		- Temperature (K).
		pressure  	- Pressure (erg/cm^3).	
	"""
	# Sanitize inputs so that only numpy arrays come in

	if not hasattr(temp, "__len__"):
		temp = [temp]
	if not hasattr(pressure, "__len__"):
		pressure = [pressure]

	temp = np.array(temp)
	pressure = np.array(pressure)

	maxLen = max(len(temp),len(pressure))
	if maxLen > 1:
		if len(temp) == 1:
			temp = temp[0]*np.ones(maxLen)
		if len(den) == 1:
			pressure = pressure[0]*np.ones(maxLen)

	return temp,pressure

class scvh:

	def __init__(self,dirname='SourceTables/SCVH/'):
		self.dirname = dirname

		# Read in the hydrogen EOS table
		hTable,logTvals,logPvals = readTables(dirname+'H_TAB_I.DAT')

		# We store the tables as a dictionary, with named 2D tables corresponding to the stored variables
		self.Hnames = ['XH2','XH','logRho','logS','logU','dLogRho/dLogT|P','dLogRho/dLogP|T','dLogS/dLogT|P','dLogS/dLogP|T','dLogT/dLogP|S']
		self.hTables = {self.Hnames[i]:hTable[:,:,i] for i in range(len(self.Hnames))}
		self.hLogTvals = logTvals
		self.hLogPvals = logPvals

		# Read in the helium EOS table
		hTable,logTvals,logPvals = readTables(dirname+'HE_TAB_I.DAT')

		# We store the tables as a dictionary, with named 2D tables corresponding to the stored variables
		self.HeNames = ['XHe','XHe+','logRho','logS','logU','dLogRho/dLogT|P','dLogRho/dLogP|T','dLogS/dLogT|P','dLogS/dLogP|T','dLogT/dLogP|S']
		self.HeTables = self.hTable
#		{self.HeNames[i]:hTable[:,:,i] for i in range(len(self.HeNames))}
		self.HeLogTvals = logTvals
		self.HeLogPvals = logPvals

		# Read in the hydrogen phase transition data
		rho_crit = np.transpose(np.loadtxt(dirname+'RHO_CRIT.DAT'))

		# We store the tables as a dictionary, with named 2D tables corresponding to the stored variables
		self.critNames = ['logT','logP','logRho1','logRho2']		
		self.rhoCrit = {self.critNames[i]:rho_crit[i] for i in range(len(self.critNames))}

		# Read in the Phase 1 transition data
		hTable,logTvals,logPvals = readTables(dirname+'H_TAB_P1.DAT')

		# We store the tables as a dictionary, with named 2D tables corresponding to the stored variables
		self.P1Tables = {self.Hnames[i]:hTable[:,:,i] for i in range(len(self.Hnames))}
		self.P1LogTvals = logTvals
		self.P1LogPvals = logPvals

		# Read in the Phase 2 transition data
		hTable,logTvals,logPvals = readTables(dirname+'H_TAB_P2.DAT')

		# We store the tables as a dictionary, with named 2D tables corresponding to the stored variables
		self.P2Tables = {self.Hnames[i]:hTable[:,:,i] for i in range(len(self.Hnames))}
		self.P2LogTvals = logTvals
		self.P2LogPvals = logPvals

		# Need to 

		# Create interpolator for He
		heMod = np.copy(self.HeTables)
		heMod[np.isnan(heMod)] = 0 # Only okay because the NaN values appear on the edges
		self.HeMask = 1.0*np.isnan(heMod)
		self.HeInterp = RectBivariateSpline(self.HeLogTvals,self.HeLogPvals,heMod,kx=3,ky=3)
		self.HeMaskInterp = RectBivariateSpline(self.HeLogTvals,self.HeLogPvals,self.HeMask,kx=3,ky=3)

		# Create interpolator for H
		hMod = np.copy(self.hTables)
		hMod[np.isnan(hMod)] = 0 # Only okay because the NaN values appear on the edges
		self.HMask = 1.0*np.isnan(hMod)
		self.HInterp = RectBivariateSpline(self.hLogTvals,self.hLogPvals,hMod,kx=3,ky=3)
		self.HMaskInterp = RectBivariateSpline(self.hLogTvals,self.hLogPvals,self.HMask,kx=3,ky=3)

		# Create interpolator for H (Phase transition 1)
		hMod = np.copy(self.P1Tables)
		hMod[np.isnan(hMod)] = 0 # Only okay because the NaN values appear on the edges
		self.P1Mask = 1.0*np.isnan(hMod)
		self.P1Interp = RectBivariateSpline(self.P1LogTvals,self.P1LogPvals,hMod,kx=3,ky=3)
		self.P1MaskInterp = RectBivariateSpline(self.P1LogTvals,self.P1LogPvals,self.P1Mask,kx=3,ky=3)

		# Create interpolator for H (Phase transition 2)
		hMod = np.copy(self.P2Tables)
		hMod[np.isnan] = 0 # Only okay because the NaN values appear on the edges
		self.P2Mask = 1.0*np.isnan(hMod)
		self.P2Interp = RectBivariateSpline(self.P2LogTvals,self.P2LogPvals,hMod,kx=3,ky=3)
		self.P2MaskInterp = RectBivariateSpline(self.P2LogTvals,self.P2LogPvals,self.P2Mask,kx=3,ky=3)

		# Create transition interpolator
		self.criticalLineInterpolator = interp1d(self.rhoCrit['logT'],self.rhoCrit['logP'],kind='linear',bounds_error=False,fill_value=np.nan)

	def interpolateHe(self,temp,pressure):
		"""
		Computes the equation of state for He and returns the various thermodynamic quantities.

		Arguments:
		temp 		- Temperature (K).
		pressure  	- Pressure (erg/cm^3).

		TODO: Document outputs

		All arguments should either be floats or 1-dimensional numpy arrays. Mixed inputs
		between floats and arrays are allowed so long as each array present has the same length.
		The return value will either be a one-dimensional vector (if all inputs are floats),
		or an 2-dimensional vector (otherwise).
		"""

		temp, pressure = sanitizeInput(temp, pressure)
		out = self.HeInterp(temp,pressure)
		outMask = self.HeMaskInterp(temp,pressure)
		out[outMask < 0.5] = np.nan # If the mask falls this low we are outside the grid.

		return out

	def interpolateSmoothedH(self,temp,pressure):
		"""
		Computes the transition-smoothed equation of state for H and returns the various thermodynamic quantities.

		Arguments:
		temp 		- Temperature (K).
		pressure  	- Pressure (erg/cm^3).

		TODO: Document outputs

		All arguments should either be floats or 1-dimensional numpy arrays. Mixed inputs
		between floats and arrays are allowed so long as each array present has the same length.
		The return value will either be a one-dimensional vector (if all inputs are floats),
		or an 2-dimensional vector (otherwise).
		"""

		temp, pressure = sanitizeInput(temp, pressure)
		out = self.HInterp(temp,pressure)
		outMask = self.HMaskInterp(temp,pressure)
		out[outMask < 0.5] = np.nan # If the mask falls this low we are outside the grid.

		return out

	def interpolateH(self,temp,pressure):
		"""
		Computes the equation of state for H including the plasma phase transition
		and returns the various thermodynamic quantities.

		Arguments:
		temp 		- Temperature (K).
		pressure  	- Pressure (erg/cm^3).

		TODO: Document outputs

		All arguments should either be floats or 1-dimensional numpy arrays. Mixed inputs
		between floats and arrays are allowed so long as each array present has the same length.
		The return value will either be a one-dimensional vector (if all inputs are floats),
		or an 2-dimensional vector (otherwise).
		"""

		temp, pressure = sanitizeInput(temp, pressure)
		logT = np.log10(temp)
		logP = np.log10(pressure)
		out = self.interpolateSmoothedH(temp,pressure)

		whereP1 = np.where((logT>4.18) & (logT>3.54) & (logT<4.82) & (logP>10.5) & (logP<14.1))
		whereP2 = np.where((logT<=4.18) & (logT>3.54) & (logT<4.82) & (logP>10.5) & (logP<14.1))
		outP1 = self.P1Interp(temp,pressure)
		outP2 = self.P2Interp(temp,pressure)
		outMask1 = self.P1MaskInterp(temp,pressure)
		outMask2 = self.P2MaskInterp(temp,pressure)
		outP1[outMask1 < 0.5] = np.nan # If the mask falls this low we are outside the grid.
		outP2[outMask2 < 0.5] = np.nan # If the mask falls this low we are outside the grid.
		pcrit = self.criticalLineInterpolator(np.log10(temp))
		pcrit[logT>4.18] = 11.75
		out[whereP1] = outP1[whereP1]
		out[whereP2] = outP1[whereP2]
		return out











