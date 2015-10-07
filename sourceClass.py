from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import gaussian_filter
import numpy as np

class source:
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
			['P','gradad','gamma1'] for the pressure, adiabatic gradient, and first adiabatic
			index.
	data 	-- The data table of interest. The number of dimensions this array contains must be
			one plus the size of the first dimension of grid (i.e. grid.shape[0]+1). The
			final dimension indexes the quantities of interest. The entries should be NaN when
			data is unavailable.
	linear 	-- Boolean specifying if the interpolation should be linear or not. Default is true.
			THIS MUST BE SET TO TRUE FOR INTERPOLATION IN D>2. If False, the interpolation defaults
			to bicubic splines.
	smoothingDist -- This gives the distance over which the binary mask is smoothed in grid units. Default is 4.

	For both input and output, a full specification of the available strings and the quantities
	they correspond to is given in the file 'VariableDeclarations.md' in the top level of the
	repository.

	In addition to providing interpolation routines, the source object also computes two mask
	arrays. The first, called binaryMask, is one if the requested point lies inside the
	available data and zero otherwise. The second, called smoothMask, specifies how far any
	individual position is from the edges of the available data. This mask is zero by definition
	where data is unavailable, and should quickly approach unity as one moves away from the edges
	of the data. This mask should smoothly vary as a function of position in the tables. At
	the edges (i.e. the outermost points with defined values) it should take on the a small but
	non-zero value. This value should be consistent (in terms of the function used to blur the
	binary mask into the smooth one) with the mask reaching zero at the next point (i.e. the
	first one outside of the data range).

	The purpose of the smooth mask object is to make it easy to blend multiple source objects
	together. The output of multiple sources may simply be averaged, with weights given by
	their masks. These are guaranteed by the properties of the masks to vary smoothly, even as
	the point of interest moves into regions covered by a distinct set of tables, and to only
	include tables which have data relevant to the point of interest.
	"""

	def __init__(self, grid, inNames, outNames, data, linear=True, binaryMask=None, smoothingDist=4):

		self.grid = np.copy(grid)
		self.inNames = inNames
		self.outNames = outNames
		self.data =  np.copy(data)
		self.linear = linear

		if binaryMask is None:
			# Construct the binary mask
			self.binaryMask = 1-1.0*np.isnan(data[...,0])
		else:
			self.binaryMask = np.copy(binaryMask)

		# Construct the smooth mask
		x = np.copy(self.binaryMask)
		# This procedure makes it smoothly drop off towards the edges, while being precisely
		# zero once out of bounds.
		for i in range(smoothingDist): 
			x = gaussian_filter(x,sigma=1,order=0,mode='constant',cval=0)*x
		self.smoothMask = x/np.amax(x) # Ensures that smooth masks are equally weighted across sources

		# Construct interpolators
		if linear:
			self.interpolator = RegularGridInterpolator(grid,data,bounds_error=False, fill_value=0)
		else:
			data = np.copy(data)
			data[np.isnan(data)] = 0
			self.interpolator = [RectBivariateSpline(grid[0],grid[1],data[...,i],kx=3,ky=3) for i in range(data.shape[-1])]

		# The mask interpolators are always linear. This ensures that we don't get oscillation, negative values, or other ill-conditioning.
		self.binaryMaskInterpolataor = RegularGridInterpolator(grid,self.binaryMask,bounds_error=False, fill_value=0)
		self.smoothMaskInterpolataor = RegularGridInterpolator(grid,self.smoothMask,bounds_error=False, fill_value=0)


	def namedInterp(self,points,name):
		"""
		Interpolates the tables and both masks to the specified point, and returns the named quantity.

		Arguments:
		points -- A numpy array of dimension one or two. If one, it must have size grid.shape[0].
				  If two it must have shape (N,grid.shape[0]), where N is the number of points
				  specified.
		name   -- A string matching one of the strings in outNames. This specifies the quantity to
				  output.
		"""
		ind = self.outNames.index(name)
		if ind == -1:
			print 'Error: Requested name not available from this source.'
			exit()
		if self.linear:
			return self.interpolator(points)[ind],self.binaryMaskInterpolataor(points),self.smoothMaskInterpolataor(points)
		else:
			return self.interpolator[ind](points[...,0],points[...,1],grid=False),self.binaryMaskInterpolataor(points),self.smoothMaskInterpolataor(points)

	def interp(self,points):
		"""
		Interpolates the tables and both masks to the specified point, and returns all outputs.

		Arguments:
		points -- A numpy array of dimension one or two. If one, it must have size grid.shape[0].
				  If two it must have shape (N,grid.shape[0]), where N is the number of points
				  specified.
		"""
		if self.linear:
			return self.interpolator(points),self.binaryMaskInterpolataor(points),self.smoothMaskInterpolataor(points)
		else:
			return self.interpolator(points[...,0],points[...,1],grid=False),self.binaryMaskInterpolataor(points),self.smoothMaskInterpolataor(points)

def overlapNamedSources(sources, points, name, weights=None):
	"""
	Merges the output from multiple sources.

	Arguments:
	sources -- The source objects to merge.
	points  -- The points at which to interpolate.
	name 	-- The quantity to interpolate.
	weights -- The weighting of the various sources, given as a numpy array of length len(sources).
			   Default is None, which is processed as np.ones(len(sources)).

	The sources are averaged, weighted based on their smoothMask values.
	These values are zero out-of-bounds, and unity in the interior of the data set.
	At the edges of the data set, the mask smoothly transitions between 1 and 0.
	"""
	if weights is None:
		weights = np.ones(len(sources))

	output = []
	masks = []
	for s in sources:
		out, _, smoothMask = s.namedInterp(points,name)
		output.append(out)
		masks.append(smoothMask)
	output = np.array(output)
	masks = np.array(masks)*weights[:,np.newaxis]
	norm = np.sum(masks,axis=0)
	out = np.sum(output*masks/norm,axis=0)
	return out







