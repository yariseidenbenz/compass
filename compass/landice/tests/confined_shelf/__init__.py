from compass.testgroup import TestGroup
from compass.landice.tests.confined_shelf.decomposition_test \
    import DecompositionTest


class ConfinedShelf(TestGroup):
    """
    A test group for confined shelf test cases
    See http://homepages.vub.ac.be/~phuybrec/eismint/shelf-descr.pdf
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='confined_shelf')

        for mesh_type in ['5000m', ]:
            self.add_test_case(DecompositionTest(test_group=self,
                               mesh_type=mesh_type))
