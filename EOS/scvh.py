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

import sys
sys.path.append('../')
from sourceClass import *

def readTables(fname):
	"""
	This function reads in any of the SCVH tables (other than the critical line table),
	given the file name as an argument. It returns the table, the grid values of
	log T (Temperature in K), and the grid values of log P (Pressure in erg/cm^3).
	Missing entries are filled with NaN. The structure of the tables guarantees that
	these entries are always grouped at the high-pressure end, and that for any
	(T,P1) giving NaN, (T,P2>P1) gives NaN too.

	The tables themselves are indexed first by T, then by P, and finally by the quantities
	of interest. For hydrogen tables, this last column is given as:
	0 - Number fraction of H2
	1 - Number fraction of H
	2 - Log(Density (g/cm^3))
	3 - Log(Entropy (erg/K/g))
	4 - Log(Internal energy (erg/g))
	5 - dLogRho/dLogT|P
	6 - dLogRho/dLogP|T
	7 - dLogS/dLogT|P
	8 - dLogS/dLogP|T
	9 - dLogT/dLogP|S

	For helium, they are:
	0 - Number fraction of He
	1 - Number fraction of He+ ions
	2 - Log(Density (g/cm^3))
	3 - Log(Entropy (erg/K/g))
	4 - Log(Internal energy (erg/g))
	5 - dLogRho/dLogT|P
	6 - dLogRho/dLogP|T
	7 - dLogS/dLogT|P
	8 - dLogS/dLogP|T
	9 - dLogT/dLogP|S

	"""
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

def makeSources(dirname='SourceTables/SCVH/'):
	# Read in the smoothed hydrogen EOS table and construct the source
	hTable,logTvals,logPvals = readTables(dirname+'H_TAB_I.DAT')
	outNames = ['X(H2)','X(H)','logRho','logS','logU','dLogRho/dLogT|P','dLogRho/dLogP|T','dLogS/dLogT|P','dLogS/dLogP|T','dLogT/dLogP|S']
	smoothedH = source([logTvals,logPvals],['logT','logP'],outNames,hTable,linear=False)

	# Read in the helium EOS table and construct the source
	heTable,logTvals,logPvals = readTables(dirname+'HE_TAB_I.DAT')
	outNames = ['X(He)','X(He+)','logRho','logS','logU','dLogRho/dLogT|P','dLogRho/dLogP|T','dLogS/dLogT|P','dLogS/dLogP|T','dLogT/dLogP|S']
	He = source([logTvals,logPvals],['logT','logP'],outNames,heTable,linear=False)

	# Read in the hydrogen phase transition data
	rhoCrit = np.transpose(np.loadtxt(dirname+'RHO_CRIT.DAT'))

	# Read in the Phase 1 transition data
	hTable1,logTvals1,logPvals1 = readTables(dirname+'H_TAB_P1.DAT')
	outNames = ['X(H2)','X(H)','logRho','logS','logU','dLogRho/dLogT|P','dLogRho/dLogP|T','dLogS/dLogT|P','dLogS/dLogP|T','dLogT/dLogP|S']

	# Read in the Phase 2 transition data
	hTable2,logTvals2,logPvals2 = readTables(dirname+'H_TAB_P2.DAT')
	outNames = ['X(H2)','X(H)','logRho','logS','logU','dLogRho/dLogT|P','dLogRho/dLogP|T','dLogS/dLogT|P','dLogS/dLogP|T','dLogT/dLogP|S']

	# Create the binary masks for the phase transition

		# Create interpolator for the critical line
	criticalLineInterpolator = interp1d(rhoCrit[0],rhoCrit[1],kind='linear',bounds_error=False,fill_value=np.nan)
		# Interpolate the critical line in T for Phase 1
	critP1 = criticalLineInterpolator(logTvals1)
		# Zero-out the mask where it falls above the critical pressure
	binaryMask1 = np.ones((len(logTvals1),len(logPvals1)))
	for i in range(len(logTvals1)):
		binaryMask1[i,logPvals1>critP1[i]] = 0

		# Interpolate the critical line in T for Phase 2
	critP2 = criticalLineInterpolator(logTvals1)
		# Zero-out the mask where it falls below the critical pressure
	binaryMask2 = np.ones((len(logTvals2),len(logPvals2)))
	for i in range(len(logTvals2)):
		binaryMask2[i,logPvals2>critP2[i]] = 0

	# Create the transition sources
	HP1 = source([logTvals1,logPvals1],['logT','logP'],outNames,hTable1,linear=False,binaryMask=binaryMask1,smoothingDist=7)
	HP2 = source([logTvals2,logPvals2],['logT','logP'],outNames,hTable2,linear=False,binaryMask=binaryMask2,smoothingDist=7)

	return smoothedH,HP1,HP2,He

def test():
	import matplotlib.pyplot as plt

	smoothedH,HP1,HP2,He = makeSources('SourceTables/SCVH/')
	t = np.linspace(min(smoothedH.grid[0]),max(smoothedH.grid[0]),num=100,endpoint=True)
	p = np.linspace(min(smoothedH.grid[1]),max(smoothedH.grid[1]),num=100,endpoint=True)
	t,p = np.meshgrid(t,p)
	t = t.flatten()
	p = p.flatten()
	data = overlapNamedSources([HP1,HP2,He],np.transpose(np.array([t,p])),'logRho')
	data = np.reshape(data,(100,100))
	plt.subplot(221)
	plt.imshow(data,extent=[min(smoothedH.grid[0]),max(smoothedH.grid[0]),min(smoothedH.grid[1]),max(smoothedH.grid[1])],origin='lower',aspect=0.4)
	plt.xlabel('log T')
	plt.ylabel('log P')
	plt.colorbar()
	plt.subplot(222)
	data = overlapNamedSources([HP1,HP2,He],np.transpose(np.array([t,p])),'logRho',weights=np.array([10,10,1]))
	data = np.reshape(data,(100,100))
	plt.imshow(data,extent=[min(smoothedH.grid[0]),max(smoothedH.grid[0]),min(smoothedH.grid[1]),max(smoothedH.grid[1])],origin='lower',aspect=0.4)
	plt.xlabel('log T')
	plt.ylabel('log P')
	plt.colorbar()
	plt.subplot(223)
	data = overlapNamedSources([HP1,HP2,He],np.transpose(np.array([t,p])),'logRho',weights=np.array([0,0,1]))
	data = np.reshape(data,(100,100))
	plt.imshow(data,extent=[min(smoothedH.grid[0]),max(smoothedH.grid[0]),min(smoothedH.grid[1]),max(smoothedH.grid[1])],origin='lower',aspect=0.4)
	plt.xlabel('log T')
	plt.ylabel('log P')
	plt.colorbar()
	plt.show()
	exit()

