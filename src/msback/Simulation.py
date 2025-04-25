import threading
import copy as cp
from msback.MSToolProtein import MSToolProtein
from msback.Utils import timer
from openmm.app import *
from openmm import *
from openmm.unit import *




class QuickSim(object):
    def __init__(self, name, device=None):

        self.name = name.split(".")[0]
        pdb = PDBFile(name)
        forcefield = ForceField('charmm36.xml', 'charmm36/tip3p-pme-b.xml')
        modeller = Modeller(pdb.topology, pdb.positions)
        print('Adding hydrogens...')
        modeller.addHydrogens(forcefield)
        #print('Adding solvent...')
        top = modeller.getTopology()
        top.setUnitCellDimensions((20, 20, 20))
        chains = top.chains()
        print('Minimizing...')
        system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME)
        integrator = VerletIntegrator(0.001 * picoseconds)
        platform = Platform.getPlatformByName('OpenCL')

        properties = {'Precision': 'double'}
        if device is not None:
            properties['DeviceIndex'] = str(device)
        self.simulation = Simulation(modeller.topology, system, integrator, platform, properties)
        self.simulation.context.setPositions(modeller.positions)

        self.topology = top
        self.forcefield = forcefield
        self.modeller = modeller
        self.system = system

    @timer
    def run_min(self, max_iterations=100):
        initial_energy = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()._value
        print("E0: %.3e kJ/mol" % initial_energy)
        self.simulation.minimizeEnergy(maxIterations=max_iterations)

        final_energy = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()._value
        print("E1: %.3e kJ/mol" % final_energy)

        positions = self.simulation.context.getState(getPositions=True).getPositions()

        PDBFile.writeFile(self.simulation.topology, positions, open(self.name + '_em.pdb', 'w'))




lock = threading.Lock()





def sim_worker(name, device_index):
        try:
            qs = QuickSim(name, device=device_index)
            qs.run_min()

        except OpenMMException as o:
            print("caught openmm error, likely due to a clash")


