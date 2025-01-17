from compass.step import Step
from compass.ocean.iceshelf import adjust_ssh
from compass.ocean.tests.isomip_plus.forward import get_time_steps


class SshAdjustment(Step):
    """
    A step for iteratively adjusting the pressure from the weight of the ice
    shelf to match the sea-surface height as part of ice-shelf 2D test cases
    """
    def __init__(self, test_case, resolution, cores=1, min_cores=None,
                 threads=1):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : float
            The horizontal resolution (km) of the test case

        cores : int, optional
            the number of cores the step would ideally use.  If fewer cores
            are available on the system, the step will run on all available
            cores as long as this is not below ``min_cores``

        min_cores : int, optional
            the number of cores the step requires.  If the system has fewer
            than this number of cores, the step will fail

        threads : int, optional
            the number of threads the step will use

        """
        if min_cores is None:
            min_cores = cores
        super().__init__(test_case=test_case, name='ssh_adjustment',
                         cores=cores, min_cores=min_cores, threads=threads)

        # generate the namelist, replacing a few default options
        # start with the same namelist settings as the forward run
        self.add_namelist_file('compass.ocean.tests.isomip_plus',
                               'namelist.forward_and_ssh_adjust')

        # we don't want the global stats AM for this run
        options = get_time_steps(resolution)
        options['config_AM_globalStats_enable'] = '.false.'
        self.add_namelist_options(options)

        # we want a shorter run and no freshwater fluxes under the ice shelf
        # from these namelist options
        self.add_namelist_file('compass.ocean.namelists',
                               'namelist.ssh_adjust')

        self.add_streams_file('compass.ocean.streams', 'streams.ssh_adjust')

        self.add_input_file(filename='adjusting_init0.nc',
                            target='../initial_state/initial_state.nc')

        self.add_input_file(filename='graph.info',
                            target='../initial_state/culled_graph.info')

        self.add_model_as_input()

        self.add_output_file(filename='adjusted_init.nc')

    # no setup() is needed

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        iteration_count = config.getint('ssh_adjustment', 'iterations')
        adjust_ssh(variable='landIcePressure', iteration_count=iteration_count,
                   step=self)
