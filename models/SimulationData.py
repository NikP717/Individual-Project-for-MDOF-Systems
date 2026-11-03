
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
import polars as pl
import time

PARAMETERS = ["M_kg", "C_Nsm", "K_Nm", "RMS_KE_J"]

@dataclass
class DataSample():
    mass_array: list 
    damping_array: list
    spring_array: list
    rms_kinetic_energy_array: list

    def __post_init__(self):
        # prevent modifying from indirect external effects
        self.mass_array = self.mass_array.copy()
        self.damping_array = self.damping_array.copy()
        self.spring_array = self.spring_array.copy()

    def to_row(self):
        return self.mass_array + self.damping_array + self.spring_array + self.rms_kinetic_energy_array

class Dataset():
    def __init__(self, batch_name: str):
        self.name = batch_name
        self.dataset = []
        self.columns = None
        self.buffer_file_number = 1
        self.buffer_limit = 1000000

        self.data_dir = self._init_dir()

    def add_data_row(self, data: DataSample) -> None:
        if self.columns is None:
            self._init_dataset(data)
        self.dataset.append(data.to_row())

        if len(self.dataset) >= self.buffer_limit:
            self._save_dataset_segment()

    def save(self):
        if self.dataset:
            self._save_dataset_segment()

    @classmethod
    def load_entire_dataframe(cls, name: str, columns: list = None) -> pd.DataFrame:
        print("loading")
        time_0 = time.perf_counter()
        project_dir = Path(__file__).resolve().parent.parent
        data_dir = project_dir / "DATA" / name
        if columns is None:
            return pl.scan_parquet(str(data_dir / "*.parquet")).collect().to_pandas()
        a = pl.scan_parquet(str(data_dir / "*.parquet"))
        print("scan done")
        print(f"{time.perf_counter()-time_0:.2f} s")
        a=a.select(columns)
        print("columns done")
        print(f"{time.perf_counter()-time_0:.2f} s")
        a= a.collect()
        print("collection done (lazyframe)")
        print(f"{time.perf_counter()-time_0:.2f} s")
        a=a.to_pandas()
        print("pandas conversion done")
        print(f"{time.perf_counter()-time_0:.2f} s")

    @classmethod
    def dataset_to_excel(cls, name: str, dataset: pd.DataFrame) -> None:
        project_dir = Path(__file__).resolve().parent.parent
        data_dir = project_dir / "DATA" / name
        dataset.to_excel(data_dir / f"{name}.xlsx")

    def _save_dataset_segment(self):
        folder = self.data_dir / self.name
        folder.mkdir(parents=True, exist_ok=True)
        file = folder / f"{self.buffer_file_number:08d}.parquet"
        self.buffer_file_number += 1
        self.dataset = pd.DataFrame(self.dataset, columns = self.columns)
        self.dataset.to_parquet(file)
        self.dataset = []

    def _init_dataset(self, data:DataSample) -> None:
        columns = []
        dof = len(data.mass_array)
        for parameter in PARAMETERS:
            for i in range(1,dof+1):
                columns.append(f"{parameter}_{i}")
        self.columns = columns

    def _init_dir(self):
        project_dir = Path(__file__).resolve().parent.parent
        data_dir = project_dir / "DATA"
        return data_dir
    
