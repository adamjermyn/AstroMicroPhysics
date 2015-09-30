import numpy as np
import os

def parseHHe(fname='SourceTables/OPAL2005/EOS5_data_H-He'):
	"""
	Parses the OPAL2005 H-HE EOS tables.

	Keyword Argument:
	fname -- String giving the path of tables (default is 'SourceTables/OPAL2005/EOS5_data_H-He')

	This method parses the OPAL2005 EOS tables, excluding the H-He table.
	It returns, in order, the X values covered, the
	density values covered, the temperature values covered, and the full tables.
	The tables are organized as a four-dimensional NumPy array.
	This array is indexed as X, Density, Temperature, Quantity of Interest.
	The Quantities of Interest in the returned table (according to index) are:
	0. Mean particle mass (in proton masses)
	1. Log[Electron Number Density (1/cm^3)]
	2. Pressure (erg/cm^3)
	3. Specific energy, referenced to T=1e6K (erg/g)
	4. Specific entropy (erg/g/K)
	5. dE/dRho|T (erg*cm^3/g^2)
	6. Specific heat at constant volume (erg/g/K)
	7. ChiR (d Log P/d Log Rho)|T (Dimensionless)
	8. ChiT (d Log P/d Log T)|Rho (Dimensionless)
	9. Gamma1 (Dimensionless)
	10. Gamma2/(Gamma2-Gamma1) (Dimensionless)

	Where data are unavailable due to the tables not being rectangular, NaN is filled instead.

	The table which is parsed (as opposed to what is returned)
	is documented in the README.txt file in the source tables.
	"""

	fi = open(fname)

	# We know that Z=0, so the table is just (X,Rho,T)
	xVals = []
	# These are the rho and T values covered by the tables.
	rhoVals = set()
	tVals = set()
	# The tables themselves:
	tables = []

	# Parse!
	for line in fi:
		if 'X' in line: # Parse X,Z from the file
			s = line.replace('=', ' ').split(' ')
			s = [i for i in s if i != '']
			xVals.append(float(s[1]))
			tables.append([])
		elif 'density' in line: # Parse the density of the upcoming block
			s = line.replace('=', ' ').split(' ')
			s = [i for i in s if i != '']
			rhoVals.add(float(s[5]))
			tables[-1].append([])
		# Parse the data from the density block
		elif any(c.isdigit() for c in line) and not any(c.isalpha() and c!='E' for c in line): # Eliminate empty lines and non-data lines
			line = line.rstrip('\n')  # remove newlines
			s = [i for i in line.split(' ') if i != '']
			if len(s) > 2: # remove end-of-file marker (just two zeros)
				# This table is fixed column width. In some places, the numbers run into each other.
				# This only happens when a negative number appears, so we examine each line for
				# occurrences of the form 'number-number'.
				for i in range(len(s)):
					n = s[i].find('-',1)
					if n != -1 and s[i][n-1]!='E': # Found a minus sign after the first digit and not in the exponent
						a = s[i][:n]
						b = s[i][n:]
						s[i] = a
						s.insert(i+1,b)
				tables[-1][-1].append(map(float,s))
				tVals.add(tables[-1][-1][-1][0])
	# Order the rho and T values as they appear in the tables.
	# Here we make use of the fact that T is always listed in descending order in the tables,
	# while rho is always listed in ascending order.
	# We will later have to reverse the T ordering in tables.
	rhoVals = list(rhoVals)
	tVals = list(tVals)
	rhoVals.sort()
	tVals.sort()

	# Augment missing T values with NaN
	for fileXZ in tables:
		for densityBlock in fileXZ:
			while len(densityBlock) < len(tVals):
				densityBlock.append([np.nan for i in densityBlock[-1]])

	# Put everythig into numpy
	rhoVals = np.array(rhoVals)
	tVals = np.array(tVals)
	tables = np.array(tables)

	# Reverse along temperature axis
	tables = tables[:,:,::-1]

	# Remove temperature from quantities of interest:
	tables = tables[:,:,:,1:]

	# Put everything in the correct units:
	tVals *= 1e6 			# Temperature (T6 -> K)
	tables[:,:,:,3]*=1e12 	# Pressure (MBar -> erg/cm^3)
	tables[:,:,:,6]*=1e12 	# dE/dRho|T (E is in 1e12erg/g, rho is in g/cm^3, so
							# we're going 1e12erg*cm^3/g^2 -> erg*cm^3/g^2)
	tables[:,:,:,7]*=1e6	# Specific heat at constant volume
							# (1e12 erg/g/1e6 K -> erg/g/K)
	# All of the other entries are dimensionless.	

	return xVals,rhoVals,tVals,tables


def parse(dirname='SourceTables/OPAL2005'):
	"""
	Parses the OPAL2005 EOS tables, excluding the H-He table.

	Keyword Argument:
	dirname -- String giving the path of the directory containing the tables (default is 'SourceTables/OPAL2005')

	This method parses the OPAL2005 EOS tables, excluding the H-He table.
	It returns, in order, the X values covered, the Z values covered, the
	density values covered, the temperature values covered, and the full tables.
	The tables are organized as a five-dimensional NumPy array.
	This array is indexed as X, Z, Density, Temperature, Quantity of Interest.
	The Quantities of Interest in the returned table (according to index) are:
	0. Mean particle mass (in proton masses)
	1. Log[Electron Number Density (1/cm^3)]
	2. Pressure (erg/cm^3)
	3. Specific energy, referenced to T=1e6K (erg/g)
	4. Specific entropy (erg/g/K)
	5. dE/dRho|T (erg*cm^3/g^2)
	6. Specific heat at constant volume (erg/g/K)
	7. ChiR (d Log P/d Log Rho)|T (Dimensionless)
	8. ChiT (d Log P/d Log T)|Rho (Dimensionless)
	9. Gamma1 (Dimensionless)
	10. Gamma2/(Gamma2-Gamma1) (Dimensionless)

	Where data are unavailable due to the tables not being rectangular, NaN is filled instead.

	The table which is parsed (as opposed to what is returned)
	is documented in the README.txt file in the source tables.
	"""
	fNames = [os.path.join(dirname,f) for f in os.listdir(os.path.join(dirname)) if os.path.isfile(os.path.join(dirname,f))]
	# These are the X and Z values covered by the tables.
	xList = []
	zList = []
	# These are the rho and T values covered by the tables.
	rhoVals = set()
	tVals = set()
	# The tables themselves:
	tables = []

	# Parse!
	for f in fNames: # Open a file for a given X,Z
		if 'EOS5_0' in f:
			fi = open(f)
			for line in fi:
				if 'X' in line: # Parse X,Z from the file
					s = line.replace('=', ' ').split(' ')
					s = [i for i in s if i != '']
					xList.append(float(s[1]))
					zList.append(float(s[3]))
					tables.append([])
				elif 'density' in line: # Parse the density of the upcoming block
					s = line.replace('=', ' ').split(' ')
					s = [i for i in s if i != '']
					rhoVals.add(float(s[5]))
					tables[-1].append([])
				# Parse the data from the density block
				elif any(c.isdigit() for c in line) and not any(c.isalpha() and c!='E' for c in line): # Eliminate empty lines and non-data lines
					line = line.rstrip('\n')  # remove newlines
					s = [i for i in line.split(' ') if i != '']
					if len(s) > 2: # remove end-of-file marker (just two zeros)
						tables[-1][-1].append(map(float,s))
						tVals.add(tables[-1][-1][-1][0])
	# Order the rho and T values as they appear in the tables.
	# Here we make use of the fact that T is always listed in descending order in the tables,
	# while rho is always listed in ascending order.
	# We will later have to reverse the T ordering in tables.
	rhoVals = list(rhoVals)
	tVals = list(tVals)
	rhoVals.sort()
	tVals.sort()

	# Augment missing T values with NaN
	for fileXZ in tables:
		for densityBlock in fileXZ:
			while len(densityBlock) < len(tVals):
				densityBlock.append([np.nan for i in densityBlock[-1]])

	# xList and zList contain repeats, as they are just following the indexing of files.
	# Thus we need to create xVals and zVals, and reindex the table
	# so that X and Z are separate axes (there's no use indexing by file...).
	xVals = list(set(xList))
	zVals = list(set(zList))
	xVals.sort()
	zVals.sort()

	# Put everythig into numpy
	rhoVals = np.array(rhoVals)
	tVals = np.array(tVals)
	tables = np.array(tables)

	# Reshape tables so that X and Z are separate axes.
	newTables = np.nan*np.zeros((len(xVals),len(zVals),tables.shape[1],tables.shape[2],tables.shape[3]))
	for i in range(len(tables)):
		newTables[xVals.index(xList[i]),zVals.index(zList[i])] = tables[i]
	tables = newTables

	# Reverse along temperature axis
	tables = tables[:,:,:,::-1]

	# Remove temperature from quantities of interest:
	tables = tables[:,:,:,:,1:]

	# Put everything in the correct units:
	tVals *= 1e6 			# Temperature (T6 -> K)
	tables[:,:,:,:,3]*=1e12 # Pressure (MBar -> erg/cm^3)
	tables[:,:,:,:,6]*=1e12 # dE/dRho|T (E is in 1e12erg/g, rho is in g/cm^3, so
							# we're going 1e12erg*cm^3/g^2 -> erg*cm^3/g^2)
	tables[:,:,:,:,7]*=1e6  # Specific heat at constant volume
							# (1e12 erg/g/1e6 K -> erg/g/K)
	# All of the other entries are dimensionless.	

	return xVals,zVals,rhoVals,tVals,tables