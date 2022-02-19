
from netCDF4 import Dataset
import numpy as np
from collections import Counter

from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.logging import check_call

from compass.model import make_graph_file
from compass.step import Step


class SetupMesh(Step):
    """
    A step for creating a mesh and initial condition for confined_shelf
    test cases

    Attributes
    ----------
    mesh_type : str
        The resolution or mesh type of the test case
    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='setup_mesh')
        self.mesh_type = test_case.mesh_type

        self.add_output_file(filename='graph.info')
        self.add_output_file(filename='landice_grid.nc')

    # no setup() method is needed

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config
        section = config['confined_shelf']

        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')

        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc,
                                      nonperiodic_x=True,
                                      nonperiodic_y=True)

        write_netcdf(dsMesh, 'grid.nc')

        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, logger=logger)
        write_netcdf(dsMesh, 'mpas_grid.nc')

        levels = section.get('levels')
        args = ['create_landice_grid_from_generic_MPAS_grid.py',
                '-i', 'mpas_grid.nc',
                '-o', 'landice_grid.nc',
                '-l', levels,
                '--diri']

        check_call(args, logger)

        make_graph_file(mesh_filename='landice_grid.nc',
                        graph_filename='graph.info')

        _setup_confined_shelf_initial_conditions(config, logger,
                                                 filename='landice_grid.nc')


def _setup_confined_shelf_initial_conditions(config, logger, filename):
    """
    Add the initial condition to the given MPAS mesh file

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the step

    filename : str
        file to setup confined_shelf
    """

    # Open the file, get needed dimensions

    gridfile = Dataset(filename, 'r+')
    nCells = len(gridfile.dimensions['nCells'])
    nVertLevels = len(gridfile.dimensions['nVertLevels'])
    maxEdges = len(gridfile.dimensions['maxEdges'])
    if nVertLevels != 5:
        logger.info(f'nVerLevels in the supplied file was {str(nVertLevels)}.\n'
                    f' 5 levels are typically used with this test case.')
    # Get variables
    xCell = gridfile.variables['xCell'][:]
    yCell = gridfile.variables['yCell'][:]
    xEdge = gridfile.variables['xEdge'][:]
    yEdge = gridfile.variables['yEdge'][:]
    xVertex = gridfile.variables['xVertex'][:]
    yVertex = gridfile.variables['yVertex'][:]
    cellsOnCell = gridfile.variables['cellsOnCell'][:]

    # put the domain origin in the center of the center cell in the x-direction
    # and in the 2nd row on the y-direction
    # Only do this if it appears this has not already been done:
    # 15000 is to allow for the periodic cells to have been removed from the
    # mesh
    if xVertex[:].min() < 15000.0:
        logger.info('Shifting domain origin to center of shelf front, because '
                    'it appears that this has not yet been done.')
        unique_xs = np.array(sorted(list(set(xCell[:]))))
        # center of domain range
        targetx = (unique_xs.max() - unique_xs.min()) / 2.0 + unique_xs.min()
        best_x = unique_xs[np.absolute((unique_xs - targetx)) ==
                           np.min(np.absolute(unique_xs - targetx))][0]
        logger.info('Found a best x value to use of:' + str(best_x))

        unique_ys = np.array(sorted(list(set(yCell[:]))))
        #   print unique_ys
        best_y = unique_ys[5]  # get 6th value
        logger.info(f'Found a best y value to use of:{str(best_y)}')

        xShift = -1.0 * best_x
        yShift = -1.0 * best_y
        xCell[:] = xCell[:] + xShift
        yCell[:] = yCell[:] + yShift
        xEdge[:] = xEdge[:] + xShift
        yEdge[:] = yEdge[:] + yShift
        xVertex[:] = xVertex[:] + xShift
        yVertex[:] = yVertex[:] + yShift

        gridfile.variables['xCell'][:] = xCell[:]
        gridfile.variables['yCell'][:] = yCell[:]
        gridfile.variables['xEdge'][:] = xEdge[:]
        gridfile.variables['yEdge'][:] = yEdge[:]
        gridfile.variables['xVertex'][:] = xVertex[:]
        gridfile.variables['yVertex'][:] = yVertex[:]
    
    #   print np.array(sorted(list(set(yCell[:]))))
    
    # Make a square ice mass
    # Define square dimensions - all in meters
    L = 200000.0
    
    shelfMask = np.logical_and( 
                      np.logical_and(xCell[:] >= -L/2.0, xCell[:] <= L/2.0),
                      np.logical_and(yCell[:] >= 0.0, yCell[:] <= L))
    # now grow it by one cell
    shelfMaskWithGround = np.zeros((nCells,), dtype=np.int16)
    for c in range(nCells):
        if shelfMask[c] == 1:
            for n in range(maxEdges):
                # fortran to python indexing
                neighbor = cellsOnCell[c, n] - 1
                if neighbor >= 0:
                    shelfMaskWithGround[neighbor] = 1
    # but remove the south side extension
    shelfMaskWithGround[np.nonzero(yCell[:] < 0.0)[0]] = 0
    
    thickness = gridfile.variables['thickness'][:]
    thickness[:] = 0.0
    thickness[0, np.nonzero(shelfMaskWithGround == 1)[0]] = 500.0
    gridfile.variables['thickness'][:] = thickness[:]
    gridfile.sync()
    del thickness
    
    # flat bed at -2000 m everywhere but grounded around the edges
    bedTopography = gridfile.variables['bedTopography'][0, :]
    bedTopography[np.nonzero(shelfMask == 1)[0]] = -2000.0
    bedTopography[np.nonzero(shelfMask == 0)[0]] = -440.0
    gridfile.variables['bedTopography'][0, :] = bedTopography[:]
    gridfile.sync()
    del bedTopography
    
    # Dirichlet velocity mask
    kinbcmask = gridfile.variables['dirichletVelocityMask'][:]
    kinbcmask[:] = 0
    # kinbcmask[:, np.nonzero(shelfMask==0)[0], :] = 1
    kinbcmask[:, np.nonzero(
        np.logical_and(shelfMask == 0, shelfMaskWithGround == 1))[0], :] = 1
    # kinbcmask[:, np.nonzero(yCell[:]<0.0)[0] ] = 0
    # Need to extend this mask south by one cell so that the extended FEM mask
    # will still have the 0 velo on the edges...
    # need the 4 most common x positions.
    theSides = Counter(xCell[np.nonzero(kinbcmask[0, :])[0]]).most_common(4)
    for side in theSides:
        thesideindices = np.nonzero(np.logical_and(xCell[:] == side[0],
                                                   yCell[:] <= 0.0))[0]
        kinbcmask[:, thesideindices] = 1
    # Now mark Dirichlet everywhere outside of the "box" to prevent Albany
    # from calculating the extended cell solution there
    kinbcmask[:, xCell[:] < -L/2.0] = 1
    kinbcmask[:, xCell[:] > L/2.0] = 1
    kinbcmask[:, yCell[:] > L] = 1
    gridfile.variables['dirichletVelocityMask'][:] = kinbcmask[:]
    gridfile.sync()
    del kinbcmask
    
    # Dirichlet velocity values
    gridfile.variables['uReconstructX'][:] = 0.0
    gridfile.variables['uReconstructY'][:] = 0.0
    
    # mu is 0 everywhere (strictly speaking it should not be necessary to set this)
    gridfile.variables['muFriction'][:] = 0.0
    
    # Setup layerThicknessFractions
    gridfile.variables['layerThicknessFractions'][:] = 1.0 / nVertLevels
    
    # boundary conditions
    SMB = gridfile.variables['sfcMassBal'][:]
    SMB[:] = 0.0  # m/yr
    # Convert from units of m/yr to kg/m2/s using an assumed ice density
    SMB[:] = SMB[:] * 910.0/(3600.0*24.0*365.0)
    gridfile.variables['sfcMassBal'][:] = SMB[:]
    gridfile.sync()
    del SMB
    
    gridfile.close()
    
    logger.info(f'Successfully added confined-shelf initial conditions to:'
                f' {filename}')
