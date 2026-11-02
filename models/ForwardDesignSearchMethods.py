from abc import ABC, abstractmethod
from dataclasses import dataclass
from models.SimulationData import Dataset, DataSample
from models.DOFModel import MassSpringDamper, MassSpringDamperSystem
import time
from itertools import product
import numpy as np

@dataclass
class MethodConfig():
    dataset_name: str
    DOF: int
    wall_connection_mass: float
    wall_connection_spring: float
    wall_connection_damper: float
    wall_force: float
    k_range: tuple
    kstep: float
    c_range: tuple
    cstep: float

class BaseMethod(ABC):
    # REQUIRED PROPERTIES OF A METHOD
    data: Dataset
    config: MethodConfig
    m_array: list
    F_array: list
    k_array: list
    c_array: list
    start_time: float
    F_FQ_HZ: float

    def __init__(self, config: MethodConfig) -> None:
        """All methods must have a configuration implementation"""
        self.data = Dataset(config.dataset_name)
        self.config = config
        self.DOF = self.config.DOF
        self.m_array = [config.wall_connection_mass] * self.DOF
        self.F_array = [config.wall_force] + [0] * (self.DOF-1)
        self.k_array = [config.wall_connection_spring] * self.DOF
        self.c_array = [config.wall_connection_damper] * self.DOF
        self.FQ_HZ = 0
        
        self.start_time = 0

    @abstractmethod
    def run(self) -> None:
        """All methods must have a run initialisation, this run must utilise the following functions: 
        super().run_MSDS_configuration(MSD) -> outputs data
        super().push_data_sample(data_row) -> stores data in classes embedded data"""

    def start_timer(self) -> None:
        self.start_time = time.perf_counter()

    def run_MSDS_configuration(self, MSD: MassSpringDamper) -> DataSample:
        current_config = MassSpringDamperSystem(MSD)
        current_config.calculate_SS_system(display=False)
        data_row = DataSample(MSD.masses, MSD.dampers, MSD.springs, current_config.KE)
        if self.FQ_HZ == 0:
            self.FQ_HZ = current_config.fq_HZ
        return data_row

    def push_data_sample(self, data_row) -> None:
        self.data.add_data_row(data_row)

    def save_data(self) -> None:
        self.data.save()

    @property
    def time_taken(self) -> float:
        return time.perf_counter() - self.start_time 


class ManualParameterSweep(BaseMethod):
    def run(self) -> None:
        self.start_timer()
        k_value_array = np.arange(self.config.k_range[0],self.config.k_range[1]+self.config.kstep, self.config.kstep).tolist()
        # c_value_array = np.arange(self.config.c_range[0],self.config.c_range[1]+self.config.cstep, self.config.cstep).tolist()
        permutations = len(k_value_array)**(self.DOF-1)#*len(c_value_array)**(self.DOF-1)
        print(f"Number of permutations: {permutations}")

        for k_values in product(k_value_array, repeat=self.DOF-1):
            forcing_fq = self.FQ_HZ
            if forcing_fq == 0:
                forcing_fq = None
            # for c_values in product(c_value_array, repeat=self.DOF-1):
            self.k_array[1:] = k_values
                # self.c_array[1:] = c_values
            system = MassSpringDamper(DOF=self.DOF,masses=self.m_array,springs=self.k_array,dampers=self.c_array,forces=self.F_array, frequency_hz=forcing_fq)
            sample = self.run_MSDS_configuration(system)
            self.push_data_sample(sample)

        self.save_data()
        print(rf"Total Time: {self.time_taken:.2f} s, Time per permutation: {(self.time_taken)/permutations * 1e6:.2f} micro s")
        