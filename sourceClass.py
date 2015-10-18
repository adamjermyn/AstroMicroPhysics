from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline, interp1d
from scipy.ndimage.filters import gaussian_filter, minimum_filter
import numpy as np


class source:
	"""
	An object which wraps equation of state tables and functions.
	"""

	def __init__(self, inNames, outNames, contains, smoothMask, data):
		"""
		Arguments:
		inNames   -- A list of strings specifying the inputs required to receive an output, in the
						order in which they are required. For instance, one might give ['X','Z','Rho','T'] to
						specify hydrogen fraction, metallicity, density, and temperature as requirements.
		outNames  -- A list of strings specifying the outputs that the source may return, in the
						order in which they will be returned. Thus, for example, one might specify
						['P','gradad','gamma1'] for the pressure, adiabatic gradient, and first adiabatic
						index.
		contains  -- A function which takes as an argument a 2D numpy array of shape (N,len(inNames))
						containing points as specified by inNames, and returns a 1D numpy array of shape (N,),
						which takes on the value 1 if the point is inside the range over which the source has
						data and 0 otherwise.
		smoothMask-- A function which takes as an argument a 2D numpy array of shape (N,len(inNames))
						containing points as specified by inNames, and returns a 1D numpy array of shape (N,).
						The value of this array specifies how far any individual position is from the edges of
						the available data. This mask is zero by definition where data is unavailable, and should
						quickly approach unity as one moves away from the edges of the data. This mask should smoothly
						vary as a function of position in the tables.

						If the underlying data representation is discrete:
							At the edges (i.e. the outermost points with defined values) it should take on the a small
							but non-zero value. This should represent a smoothing of the output of contains. The returned
							value should be consistent (in terms of the smoothing kernel) with the function contains
							reaching zero at the next point (i.e. the first one outside of the data range). The smoothing
							should have distance of a few grid points.

						If the underlying data representation is continuous:
										At the edges this should reach precisely zero. The smoothing should have distance of order a
										few percent the overall range of the table.

		data      -- A function which takes as an argument a 2D numpy array of shape (N,len(inNames))
					containing points as specified by inNames, and returns a 1D numpy array of shape (N,len(outNames)).
					The value of the array should be the various output quantities evaluated at the various input points.
		"""
		self.inNames = inNames
		self.outNames = outNames
		self.contains = contains
		self.smoothMask = smoothMask
		self.data = data

	def nameIndices(self, names):
		"""
		Returns a 1D list containing the indices in outNames corresponding to the listed names.

		Arguments:
		names -- List of names appearing in outNames
		"""
		indices = []
		for n in names:
			indices.append(self.outNames.index(n))
		return indices


def mergeSources(sources, inNames, outNames, weights=None):
	"""
	Merges multiple sources together with optional weights.

	Arguments:
	sources -- The source objects to merge.
	inNames -- The input names on all of the sources. Must be the same across sources.
	outNames-- The quantities to output. All input sources must contain all requested names in outNames.
	weights -- The weighting of the various sources, given as a numpy array of length len(sources).
									   Default is None, which is processed as np.ones(len(sources)).

	The sources are averaged, weighted based on their smoothMask values multiplied by weights.
	These values are zero out-of-bounds, and unity in the interior of the data set.
	At the edges of the data set, the mask smoothly transitions between 1 and 0 to ensure continuity.
	"""
	if weights is None:
		weights = np.ones(len(sources))

	# Set up new contains function
	def contains(points):
		cont = [s.contains(points) for s in sources]
		cont = sum(cont) / len(cont)
		cont[cont > 0] = 1.
		return cont

	indices = [s.nameIndices(outNames) for s in sources]

	# Set up a new smoothMask function
	def smoothMask(points):
		masks = sum([sources[i].smoothMask(points)
                    for i in range(len(sources))])
		masks[masks > 1] = 1.
		return masks

	# Set up new data function
	def data(points):
		out = [sources[i].data(points)[:, indices[i]] for i in range(len(sources))]
		masks = [weights[i] * sources[i].smoothMask(points) for i in range(len(sources))]
		out = sum([out[i] * masks[i][:, np.newaxis]
                    for i in range(len(sources))])
		norm = sum(masks)
		out /= norm[:, np.newaxis]
		return out

	return source(inNames, outNames, contains, smoothMask, data)


def interpolateSources(sources, vals, newName, inNames, outNames, kine='linear'):
	"""
	Creates a source which interpolates along an additional axis between other sources. Useful for nested interpolation.

	Arguments:
	sources -- The source objects to merge. The 'contains' method of each source must agree at all points.
	vals    -- The values of the sources along the new axis.
	newName -- The name of the new axis.
	inNames -- The input names on all of the sources. Must be the same across sources.
	outNames-- The quantities to output. All input sources must contain all requested names in outNames.
	kind	-- String specifying the kind of interpolation. Default is 'linear'. This option is just passed
				to interp1d in scipy.interpolate, so any supported by that are supported here. 

	Returns a source object which interpolates the values output by the various input sources along the new axis,
	assigning sources to values in the new dimension given by vals. The new source object has inNames made by
	(newNames,inNames[0],...,inNames[-1]), and uses the same ordering for inputs. Note that the resulting source
	requires evaluating every source at every point. Also note that out of bounds values along the new axis
	result in 
	"""

	newInNames = [newName]
	for n in inNames:
		newInNames.append(n)

	nMin = min(vals)
	nMax = max(vals)

	def data(points):
		# Project the points into one lower dimension
		lowPoints = points[:, 1:]
		newDim = points[:, 0]
		out = np.array([sources[i].data(lowPoints)
                  for i in range(len(sources))])
		outInterpolator = interp1d(
			vals, out, axis=0, kind=kind, bounds_error=False, fill_value=0.0, assume_sorted=False)
		return outInterpolator(newDim)

	def contains(points):
		ret = sources[0].contains(points[:, 1:])
		# Filter out points which fall out of bounds along the new axis.
		ret[points[:, 0] < nMin] = 0
		ret[points[:, 0] > nMax] = 0
		return ret

	def smoothMask(points):
		ret = sources[0].smoothMask(points[:, 1:])
		# Filter out points which fall out of bounds along the new axis.
		ret[points[:, 0] < nMin] = 0
		ret[points[:, 0] > nMax] = 0
		# Smoothly drop the mask to zero at the ends.
		# Make the coordinates dimensionless
		x = (points[:, 0] - nMin) / (nMax - nMin)
		# 30 was picked so that the transition happens over ~10% of the range
		ret *= np.maximum(0, 1 - 2. / (1 + np.exp(30 * x)) -
                    2. / (1 + np.exp(30 * (1 - x))))
		return ret

	return source(newInNames, outNames, contains, smoothMask, data)


def sourceFromTables(grid, inNames, outNames, data, kind='linear', binaryMask=None, smoothingDist=4):
	"""
	An interpolator object which wraps equation of state tables.

	Arguments:
	grid 	-- Must be a NumPy array of dimension 2. The first dimension indexes the variables
									over which the table is defined. The second dimension then gives the grid points
									spanned by that variable. Thus, for example, one might give an array of the form
									[[x0,x1,x2],[z0,z1,z2,z3],[rho0,rho1,rho2,...,rhoN],[T0,T1,...,TM]], where N and
									M specify the number of rho and T values given. This imposes that the data be
									given in a regular grid.
	inNames -- A list of strings specifying the inputs required to receive an output, in the
									order in which they appear in grid. In the above example, one would give
									['X','Z','Rho','T'].
	outNames-- A list of strings specifying the outputs that the source may return, in the
									order in which they will be returned. Thus, for example, one might specify
									['P','gradad','gamma1'] for the pressure, adiabatic gradient, and first
									adiabatic index.
	data 	-- The data table of interest. The number of dimensions this array contains must be
									one plus the size of the first dimension of grid (i.e. grid.shape[0]+1). The
									final dimension indexes the quantities of interest. The entries should be NaN when
									data is unavailable.
	kind	-- String specifying the kind of interpolation. Default is 'linear'. Currently only
									'linear' and 'cubic' are supported. Note that 'linear' must be used if D>2.
	smoothingDist -- This gives the distance over which the binary mask is smoothed in grid units. Default is 4.

	For both input and output, a full specification of the available strings and the quantities
	they correspond to is given in the file 'VariableDeclarations.md' in the top level of the
	repository.

	In addition to providing interpolation routines, the source object also computes two mask
	arrays. The first, called binaryMaskTable, is one if the requested point lies inside the
	available data and zero otherwise. The second, called smoothMaskTable, specifies how far any
	individual position is from the edges of the available data. This mask is zero by definition
	where data is unavailable, and should quickly approach unity as one moves away from the edges
	of the data. This mask should smoothly vary as a function of position in the tables. At
	the edges (i.e. the outermost points with defined values) it should take on the a small but
	non-zero value. This value should be consistent (in terms of the function used to blur the
	binary mask into the smooth one) with the mask reaching zero at the next point (i.e. the
	first one outside of the data range). Interpolation routines should be provided for both
	tables. The binaryMaskTable interpolator will be the contains routine, and the smoothMaskTable
	interpolator will be the smoothMask routine. Note that the contains routine must round all
	outputs to 0 or 1, so the interpolation is only there to allow for a continuum of queries.

	The purpose of the smooth mask object is to make it easy to blend multiple source objects
	together. The output of multiple sources may simply be averaged, with weights given by
	their masks. These are guaranteed by the properties of the masks to vary smoothly, even as
	the point of interest moves into regions covered by a distinct set of tables, and to only
	include tables which have data relevant to the point of interest.
	"""

	# Construct the binaryMaskTable, if not already done.
	binaryMaskTable = None
	if binaryMask is None:
		# Construct the binary mask
		binaryMaskTable = 1 - 1.0 * np.isnan(data[..., 0])
	else:
		binaryMaskTable = np.copy(binaryMask)
	if kind == 'cubic':
		# Cubic splines use the two points on either side in each dimension. Thus we
		# need to set any point which borders a zero-mask point to zero to accomodate this.
		# This change must likewise be propagated to the smooth mask, which is why we
		# do it here. Note that the mode is constant, such that we reflect our ignorance
		# at the edges of the table. This isn't an issue for linear interpolation because
		# it only needs the neighboring points.
		binaryMaskTable = minimum_filter(binaryMaskTable, size=3, mode='constant', cval=0.0)

	# Construct the smoothMaskTable.
	x = np.copy(binaryMaskTable)
	# This procedure makes it smoothly drop off towards the edges, while being precisely
	# zero once out of bounds.
	for i in range(smoothingDist):
		x = gaussian_filter(x, sigma=1, order=0, mode='constant', cval=0) * x
	# Ensures that smooth masks are equally weighted across sources
	smoothMaskTable = x / np.amax(x)

	# Construct interpolators
	dataFunc = None
	if kind == 'linear':
		interpolator = RegularGridInterpolator(grid, data, bounds_error=False, fill_value=0)
		dataFunc = interpolator
	elif kind == 'cubic':
		data = np.copy(data)
		data[np.isnan(data)] = 0
		interpolator = [RectBivariateSpline(grid[0], grid[1], data[..., i], kx=3, ky=3) for i in range(data.shape[-1])]

		def dataFunc(points):
			out = np.zeros((len(points), len(outNames)))
			for i in range(len(outNames)):
				out[:, i] = interpolator[i](points[:, 0], points[:, 1], grid=False)
			return out
	else:
		print 'Error: Unrecognized interpolation kind.'
		exit()

	# The mask interpolators are always linear. This ensures that we don't get
	# oscillation, negative values, or other ill-conditioning.
	binaryMaskInterpolataor = RegularGridInterpolator(grid, binaryMaskTable, bounds_error=False, fill_value=0)
	smoothMaskInterpolataor = RegularGridInterpolator(grid, smoothMaskTable, bounds_error=False, fill_value=0)

	# If the binary mask interpolator gives us anything less than unity, it means that the requested point is either
	# out of range, borders a point that is out of range, or (only if nonlinear) is using an out-of-range point for
	# interpolation.
	def contains(points):
		vals = binaryMaskInterpolataor(points)
		vals[vals < 1 - 1e-10] = 0  # Give some leeway for interpolation error
		return vals

	return source(inNames, outNames, contains, smoothMaskInterpolataor, dataFunc)
