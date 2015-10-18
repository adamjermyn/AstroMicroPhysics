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
	logTvals = []  # Temperature gridpoints
	logPvals = []  # Pressure gridpoints
	hTable = []  # Hydrogen table
	maxLen = 0

	fi = open(fname)

	for line in fi:
		s = line.rstrip('\n').rstrip('\r').split(' ')
		s = [i for i in s if i != '']
		if len(s) == 2:  # Means the start of a new temperature
			logTvals.append(float(s[0]))
			hTable.append([])
		else:
			hTable[-1].append(map(float, s))
		if len(hTable[-1]) > maxLen:
			maxLen = len(hTable[-1])
			logPvals = [hTable[-1][i][0] for i in range(len(hTable[-1]))]

	# The tables always start at the same pressure, and step in the same increments,
	# but the endpoint is not uniform. For efficiency we are going to pad the table such
	# that it is rectangular.
	for i in range(len(hTable)):
		for j in range(maxLen - len(hTable[i])):
			hTable[i].append([np.nan for k in range(len(hTable[0][0]))])

	# Turn it all into NumPy arrays:
	logTvals = np.array(logTvals)
	logPvals = np.array(logPvals)
	hTable = np.array(hTable)
	hTable = hTable[..., 1:]  # Remove pressure values from table entries

	return hTable, logTvals, logPvals


def makeSources(dirname='SourceTables/SCVH/'):
	# Read in the smoothed hydrogen EOS table and construct the source
	hTable, logTvals, logPvals = readTables(dirname + 'H_TAB_I.DAT')
	outNames = ['X(H2)', 'X(H)', 'logRho', 'logS', 'logE', 'dLogRho/dLogT|P',
             'dLogRho/dLogP|T', 'dLogS/dLogT|P', 'dLogS/dLogP|T', 'dLogT/dLogP|S']
	smoothedH = sourceFromTables([logTvals, logPvals], ['logT', 'logP'],
	                             outNames, hTable, kind='cubic')

	print (hTable[:,:,-1]+hTable[:,:,-2]/hTable[:,:,-3])[::5,::5]

	# Read in the helium EOS table and construct the source
	heTable, logTvals, logPvals = readTables(dirname + 'HE_TAB_I.DAT')
	outNames = ['X(He)', 'X(He+)', 'logRho', 'logS', 'logE', 'dLogRho/dLogT|P',
             'dLogRho/dLogP|T', 'dLogS/dLogT|P', 'dLogS/dLogP|T', 'dLogT/dLogP|S']
	He = sourceFromTables([logTvals, logPvals], ['logT', 'logP'], outNames, heTable, kind='cubic')

	# Read in the hydrogen phase transition data
	rhoCrit = np.transpose(np.loadtxt(dirname + 'RHO_CRIT.DAT'))

	# Read in the Phase 1 transition data
	hTable1, logTvals1, logPvals1 = readTables(dirname + 'H_TAB_P1.DAT')
	outNames = ['X(H2)', 'X(H)', 'logRho', 'logS', 'logE', 'dLogRho/dLogT|P',
             'dLogRho/dLogP|T', 'dLogS/dLogT|P', 'dLogS/dLogP|T', 'dLogT/dLogP|S']

	# Read in the Phase 2 transition data
	hTable2, logTvals2, logPvals2 = readTables(dirname + 'H_TAB_P2.DAT')
	outNames = ['X(H2)', 'X(H)', 'logRho', 'logS', 'logE', 'dLogRho/dLogT|P',
             'dLogRho/dLogP|T', 'dLogS/dLogT|P', 'dLogS/dLogP|T', 'dLogT/dLogP|S']

	# Create the binary masks for the phase transition

	# Create interpolator for the critical line
	criticalLineInterpolator = interp1d(
		rhoCrit[0], rhoCrit[1], kind='linear', bounds_error=False, fill_value=np.nan)
	# Interpolate the critical line in T
	critP = criticalLineInterpolator(logTvals1)
	# Hard-coded value from the original code
	critP[logTvals1 > 4.18] = 11.75
	# Zero-out the mask where it falls above the critical pressure
	binaryMask1 = np.ones((len(logTvals1), len(logPvals1)))
	for i in range(len(logTvals1)):
		binaryMask1[i, logPvals1 > critP[i]] = 0

		# Zero-out the mask where it falls below the critical pressure
	binaryMask2 = np.ones((len(logTvals2), len(logPvals2)))
	for i in range(len(logTvals2)):
		binaryMask2[i, logPvals2 > critP[i]] = 0

	# Create the transition sources
	HP1 = sourceFromTables([logTvals1, logPvals1], ['logT', 'logP'], outNames,
	                       hTable1, kind='cubic', binaryMask=binaryMask1, smoothingDist=7)
	HP2 = sourceFromTables([logTvals2, logPvals2], ['logT', 'logP'], outNames,
	                       hTable2, kind='cubic', binaryMask=binaryMask2, smoothingDist=7)

	# Merge transition sources with smoothed source, weighting smoothed source
	# less so it only dominates outside of the transition region
	HP = mergeSources([HP1, HP2, smoothedH], HP1.inNames, HP1.outNames, weights=[1., 1., 0.1])

	# Now we create the mixed source as functions which perform the appropriate interpoaltion in Y.
	# At these temperatures we can neglect Z (we let X -> X+Z, and leave Y unchanged). The reason we
	# put it into H rather than He is that this is what is done in the 1995 ApJ paper in which this
	# code was presented.

	kB = 1.380650424e-16  # Boltzmann Constant (erg/K)
	mH = 1.67357e-24  # Proton mass (g)
	mHe = 6.646442e-24  # Helium mass (g)

	def Smix(xHe, xHeP, xH, xH2, y):
		# The various X's are imprecise, leading to the summation rules on either He/He+
		# or H/H2 being violated (i.e. composition fractions summing above unity).
		# This is not a big issue with linear interpolation, where the
		# maximum error matches that in the table (~1e-7), but is significant once
		# the bicubic interpolation described in the SCVH paper is used, as this
		# does not automatically conserve sums. The maximum error was estimated numerically at
		# 10%. To use these concentrations in mixing entropy calculations we need to impose
		# these rules. Suppose we have two variables, A and B, and we wish to subject them to the
		# condition A+B<=1. If the violation of this rule is not large, we can impose the
		# constraint by dividing each of A and B by some function f(A+B) obeying f(x>1)>x,
		# f(x>1) ~ x, and f(x<1) ~ 1, with f both continuous and differentiable.
		# A good function for these purposes is (1+x^n)^(1/n) fits the bill nicely, with
		# increasing sharpness as n increases. We pick n=4 arbitrarily here.
		xHe /= (1 + (xHe + xHeP)**4)**0.25
		xHeP /= (1 + (xHe + xHeP)**4)**0.25
		xH /= (1 + (xH + xH2)**4)**0.25
		xH2 /= (1 + (xH + xH2)**4)**0.25

		# Compute the electron fractions
		xHelec = (1 - xH2 - xH) / 2.
		xHeElec = (2 - 2 * xHe - xHeP) / 3.
		# Compute beta, gamma, and delta from Eq. 54,55,56 in the SCVH paper
		beta = (mH / mHe) * (y / (1 - y))
		gamma = (3. / 2) * (1 + xH + 3 * xH2) / (1 + 2 * xHe + xHeP)
		delta = beta * gamma * (3. / 2) * (2 - 2 * xHe - xHeP) / (1 - xH - xH2)

		# Compute mixing entropy
		ld = np.log(1 + delta)
		ldi = np.log(1 + 1 / delta)
		lbg = np.log(1 + beta * gamma)
		lbgi = np.log(1 + 1 / (beta * gamma))
		Smix = kB * ((1 - y) / mH) * (2 / (1 + xH + 3 * xH2)) * \
                    (lbg - xHelec * ld + beta * gamma * (lbgi - xHeElec * ldi))

		return Smix

	def dSmixd(points):
		# Points is a numpy array of shape (N,3), with the three columns being (Y, logT, logP).
		y = points[:, 0]

		# Compute the H and He outputs
		outH = HP.data(points[:, 1:])
		outHe = He.data(points[:, 1:])

		# Compute Smix
		sm = Smix(outHe[:, 0], outHe[:, 1], outH[:, 1], outH[:, 0], y)

		# Compute the H and He outputs shifted in temperature
		dlogT = np.zeros(points.shape)
		dlogT[:, 1] = -1e-10
		doutH = HP.data((points + dlogT)[:, 1:])
		doutHe = He.data((points + dlogT)[:, 1:])
		doutHneg = HP.data((points - dlogT)[:, 1:])
		doutHeneg = He.data((points - dlogT)[:, 1:])

		# Compute Smix at logT+-dlogT
		psmdt = Smix(doutHe[:, 0], doutHe[:, 1], doutH[:, 1], doutH[:, 0], y)
		msmdt = Smix(doutHeneg[:, 0], doutHeneg[:, 1], doutHneg[:, 1], doutHneg[:, 0], y)

		# Compute the H and He outputs shifted in pressure
		dlogP = np.zeros(points.shape)
		dlogP[:, 2] = -1e-10
		doutH = HP.data((points + dlogP)[:, 1:])
		doutHe = He.data((points + dlogP)[:, 1:])
		doutHneg = HP.data((points - dlogP)[:, 1:])
		doutHeneg = He.data((points - dlogP)[:, 1:])

		# Compute Smix at logP+dlogP
		psmdp = Smix(doutHe[:, 0], doutHe[:, 1], doutH[:, 1], doutH[:, 0], y)
		msmdp = Smix(doutHeneg[:, 0], doutHeneg[:, 1], doutHneg[:, 1], doutHneg[:, 0], y)

		# Compute derivative
		dsmdp = (psmdp - msmdp) / (2*dlogP[:, 2]) / sm

		# Compute derivative
		dsmdt = (psmdt - msmdt) / (2*dlogT[:, 1]) / sm

		return sm, dsmdt, dsmdp

	def data(points):

		# Points is a numpy array of shape (N,3), with the three columns being (Y, logT, logP).
		y = points[:, 0]

		# Compute the H and He outputs
		outH = HP.data(points[:, 1:])
		outHe = He.data(points[:, 1:])

		# Compute mixing entropy and derivatives
		sm, dsmdt, dsmdp = dSmixd(points)

		print sm[::30]

		out = np.zeros((len(points), 12))  # Create combined output
		out[:, 0] = (1 - y) * outH[:, 1]
		out[:, 1] = (1 - y) * outH[:, 0]
		out[:, 2] = y * outHe[:, 0]
		out[:, 3] = y * outHe[:, 1]
		out[:, 4] = -np.log10((1 - y) * 10**(-outH[:, 2]) + y * 10**(-outHe[:, 2]))  # Interpolate logRho
		out[:, 5] = np.log10((1 - y) * 10**outH[:, 3] + y * 10**outHe[:, 3] + sm)  # Interpolate logS
		out[:, 6] = np.log10((1 - y) * 10**(outH[:, 4]) + y * 10**(outHe[:, 4]))  # Interpolate logE
		out[:, 7] = (10**out[:, 4]) * ((1 - y) * outH[:, 5] * 10**(-outH[:, 2]) +
                                 y * outHe[:, 5] * 10**(-outHe[:, 2]))  # Interpolate dLogRho/dLogT
		out[:, 8] = (10**out[:, 4]) * ((1 - y) * outH[:, 6] * 10**(-outH[:, 2]) +
                                 y * outHe[:, 6] * 10**(-outHe[:, 2]))  # Interpolate dLogRho/dLogP
		out[:, 9] = 10**out[:, 5] * ((1 - y) * 10**(-outH[:, 3]) * outH[:, 7] + y * 10 **
                                        (-outHe[:, 3]) * outHe[:, 7]) + sm * 10**(-out[:, 5]) * dsmdt  # Interpolate dLogS/dLogT

		out[:, 10] = 10**out[:, 5] * ((1 - y) * 10**(-outH[:, 3]) * outH[:, 8] + y * 10 **
                                         (-outHe[:, 3]) * outHe[:, 8]) + sm * 10**(-out[:, 5]) * dsmdp  # Interpolate dLogS/dLogP
		# Below we compute dLogS/dLogP (for computing the adiabatic thermal gradient) this way rather
		# than as in the above two lines because this produces correct interpolation for dlnT/dlnP|S.
		# Computing it as above compounds interpolation error and leads to non-vanishing error in the adiabatic
		# gradient as we approach either Y=0 or Y=1 (i.e. inconsistency with the edge cases).
		# Unfortunately the system as described in SCVH is not consistent, so we cannot maintain the
		# relationship Sp/St = -dlnT/dlnP|S. Rather, we attempt to keep each quantity as consistent at the
		# boundaries of Y as possible.

		adDlogSDlogP = -10**out[:, 5] * ((1 - y) * 10**(-outH[:, 3]) * outH[:, 7]*outH[:, 9] + y * 10 **
                                         (-outHe[:, 3]) * outHe[:, 7]*outHe[:,9]) + sm * 10**(-out[:, 5]) * dsmdp  # Interpolate dLogS/dLogP
		out[:, 11] = -adDlogSDlogP / out[:, 9]

		return out

	def contains(points):
		return He.contains(points[:, 1:])

	outNames = ['X(H)', 'X(H2)', 'X(He)', 'X(He+)', 'logRho', 'logS', 'logE', 'dLogRho/dLogT|P',
             'dLogRho/dLogP|T', 'dLogS/dLogT|P', 'dLogS/dLogP|T', 'dLogT/dLogP|S']
	# The contains function we take from He, as it has the smaller range (than H) due to the lack of a pressure phase transition (we get more
	# data for H because of the detail around the transition, which occurs
	# near an edge). Likewise for the smoothMask function.
	HHe = source(['Y', 'logT', 'logP'], outNames, contains, He.smoothMask, data)

	return HHe, HP, HP1, HP1, smoothedH, He


def test():
	HHe, HP, HP1, HP1, smoothedH, He = makeSources()

	s1 = HHe
	s2 = He

	import matplotlib.pyplot as plt

	y = 1-1e-15
	t = np.linspace(2., 8., num=200)
	p = np.linspace(3., 20., num=200)
	t, p = np.meshgrid(t, p)
	t = t.flatten()
	p = p.flatten()
	points1 = np.transpose((y * np.ones(len(t)), t, p))
	points2 = np.transpose((t, p))
	out1 = s1.data(points1)
	out2 = s2.data(points2)
	out = out1[:,-2]
	outP = out2[:,-2]
	cont = s1.contains(points1)
	out[cont == 0.] = np.nan
	outP[cont == 0.] = np.nan
	cont = np.reshape(cont, (200, 200))
	out = np.reshape(out, (200, 200))
	outP = np.reshape(outP, (200, 200))
	plt.subplot(131)
	plt.imshow(out, origin='lower', extent=[2.1, 7.06, 4, 19])
	plt.colorbar()
	plt.subplot(132)
	plt.imshow(outP, origin='lower', extent=[2.1, 7.06, 4, 19])
	plt.colorbar()
	plt.subplot(133)
	plt.imshow(out - outP, origin='lower', extent=[2.1, 7.06, 4, 19])
	plt.colorbar()
	plt.show()
