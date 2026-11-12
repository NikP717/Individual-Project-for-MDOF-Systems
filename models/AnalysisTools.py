import pandas as pd
from dataclasses import dataclass
from matplotlib import pyplot as plt
from pathlib import Path
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from models.SimulationData import Dataset
from models.DOFModel import MassSpringDamper, MassSpringDamperSystem
from scipy.integrate import trapezoid
from itertools import  combinations

@dataclass
class PlotAttributes():
    data_name: str
    plot_name: str
    x_axis: str
    y_axis: str
    z_axis: str
    fixed_vals: dict = None
    plot_title: str = None
    xlabel: str = None
    ylabel: str = None
    contour_label: str = None
    logarithmic_scale: bool = False

class DataPlotter():
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def plot_data(self, attr: PlotAttributes):
        if hasattr(attr, 'z_axis'):
            if attr.z_axis is not None:
                self._contour_plot(attr)
                return
        self._2d_plot(attr)

    def _contour_plot(self, attr: PlotAttributes):
        df = self.df
        fixed_data_slice = df.iloc[0].to_dict()
        if attr.fixed_vals:
            for column, value in attr.fixed_vals.items():
                df = df[df[column] == value]                    
        Z = df.pivot(index=attr.y_axis,columns=attr.x_axis,values=attr.z_axis).sort_index().sort_index(axis=1)
        z = Z.to_numpy()
        levels = np.linspace(z.min(),1.1*z.max(),150)
        fig, ax = plt.subplots(figsize=(10, 7))
        contour = ax.contourf(Z.columns.to_numpy(),Z.index.to_numpy(),Z.to_numpy(),levels=levels,cmap="rainbow")
        fig.colorbar(contour, ax=ax, label=attr.contour_label)

        ax.set_xlabel(attr.xlabel)
        ax.set_ylabel(attr.ylabel)
        ax.set_title(attr.plot_title)
        fig.tight_layout()
        if attr.logarithmic_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")    
        self._save_plot(fig, attr)

    def _2d_plot(self, attr: PlotAttributes):
        df = self.df
        x = df[attr.x_axis].to_numpy()
        y = df[attr.y_axis].to_numpy()

        fig, ax = plt.subplots(figsize=(7,7))

        ax.set_xlabel(attr.xlabel)
        ax.set_ylabel(attr.ylabel)
        ax.set_title(attr.plot_name)
        plt.title(attr.plot_title)
        ax.grid()
        ax.plot(x, y, 'k-')
        fig.tight_layout()
        if attr.logarithmic_scale:
            ax.set_xscale("log")
        self._save_plot(fig, attr)

    def _save_plot(self, fig: plt.figure, attr: PlotAttributes):
        dir =  Path(__file__).resolve().parent.parent / "DATA" / attr.data_name
        filepath = dir / f"{attr.plot_name}.png"
        fig.savefig(filepath)
        print(f"Saved: {filepath}")
        plt.close(fig)  

class DataAnalyser():
    @classmethod
    def open_data(cls, folder_name: str, columns: list = None) -> pd.DataFrame:
        df = Dataset.load_entire_dataframe(folder_name, columns)
        return df
    
    @classmethod
    def validate_data_row(cls, dof, data_row: pd.DataFrame) -> None:
        data_row = data_row.iloc[0]
        masses, springs, dampers, forces,  system = cls._row_to_MSD(dof, data_row)
        validation_model = MassSpringDamperSystem(system)
        displacement, velocity, time = validation_model.calculate_dynamic_system((0,200))
        for i in range(1,dof+1): # max dof work down to minimum
            plt.plot(time[:2000], displacement[dof-i, :2000]*1000,label=f"Mass {i} ({masses[dof-i]}kg): k = {springs[dof-i]}, c = {dampers[dof-i]}")
        plt.legend()
        plt.title(f"Force 1: {forces[0]} N, Excitation FQ: {validation_model.fq_HZ:.3f} Hz")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement (mm)")
        plt.grid()

        # Extract the final 10 forcing cycles for last mass
        T = 1 / validation_model.fq_HZ
        mask = time >= time[-1] - 10*T

        t_ss = time[mask]
        v_ss = velocity[:, mask]
        average_KE = np.array(0.5 * masses[dof-1] *trapezoid(v_ss[dof-1]**2, t_ss) /(t_ss[-1] - t_ss[0]))
        print(f"Springs: {springs}, Dampers: {dampers}")
        print(f"Dynamic SS KE: {average_KE:.3f} J // FQ Domain analysis SS KE: {data_row[f"RMS_KE_J_{dof}"]:.3f}")
        print(f"% difference: {(average_KE - data_row[f"RMS_KE_J_{dof}"]) / data_row[f"RMS_KE_J_{dof}"]:.5f} %")
        print("---")

    @classmethod
    def locate_data_maxima(cls, data: pd.DataFrame, data_column: str) -> pd.DataFrame:
        max_row = data.loc[data[data_column][data[data_column] > 0].idxmax()]
        return max_row
    
    @classmethod
    def _row_to_MSD(cls, dof, row):
        mass_columns = row.index.str.startswith("M_kg")
        masses = row[mass_columns].to_list()
        spring_columns = row.index.str.startswith("K_Nm")
        springs = row.loc[spring_columns].to_list()
        damping_columns = row.index.str.startswith("C_Nsm")
        dampers = row.loc[damping_columns].to_list()
        forces = np.zeros_like(dampers).tolist()
        forces[0] = 1
        system = MassSpringDamper(dof, masses, springs, dampers, forces, None)
        return masses, springs, dampers, forces, system

    @classmethod
    def generate_receptance_plot(cls, dof, data_name: str) -> None:
        data = cls.open_data(data_name).copy()
        force = 1  # NOTE: HARD CODED VALUE
        c_col = f"C_Nsm_{dof}"
        k_col = f"K_Nm_{dof}"
        spring_norm_factor = data.iloc[0]['K_Nm_1']
        damp_norm_factor = data.iloc[0]['C_Nsm_1']
        # Normalise columns
        norm_k_cols = [c for c in data.columns if c.startswith("K_Nm")]
        norm_c_cols = [c for c in data.columns if c.startswith("C_Nsm")]
        data[norm_k_cols] = data[norm_k_cols] / spring_norm_factor
        data[norm_c_cols] = data[norm_c_cols] / damp_norm_factor
        # Lock every other parameter column to its value in the first row
        param_cols = [col for col in data.columns if col.startswith(("M_kg_", "C_Nsm_", "K_Nm_"))]
        mask = np.ones(len(data), dtype=bool)
        for col in param_cols:
            if col in (c_col, k_col):
                continue
            mask &= np.isclose(data[col], data[col].iloc[0])
        data = data[mask].copy()

        data["R_real"] = data[f"X_real_m_{dof}"] / force
        data["R_imag"] = data[f"X_imag_m_{dof}"] / force

        fig, ax = plt.subplots()

        # Lines of constant C (K varies along each line)
        for c, g in data.groupby(c_col):
            g = g.sort_values(k_col)
            ax.plot(g["R_real"], g["R_imag"], "-", lw=0.8, color="black")

        # Lines of constant K (C varies along each line)
        for k, g in data.groupby(k_col):
            g = g.sort_values(c_col)
            ax.plot(g["R_real"], g["R_imag"], "--", lw=0.8, color="red")

        # Proxy artists so the legend distinguishes the two families
        ax.plot([], [], "-", color="black", label=f"Constant {cls._column_to_label_translator(c_col)}")
        ax.plot([], [], "--", color="red", label=f"Constant {cls._column_to_label_translator(k_col)}")
        ax.legend()

        ax.set_aspect("equal")
        ax.set_xlabel("Re(R) (Where F1=1N)")
        ax.set_ylabel("Im(R) (Where F1=1N)")
        ax.set_title("Receptance for 2DOF System: F1 = 1N, M1&2 = 1kg, K1 = 100N/m , C1 = 1 Ns/m")
        plot_name = "Receptance Plot with varied K2 and C2 for 2DOF System"
        dir = Path(__file__).resolve().parent.parent / "DATA" / data_name
        filepath = dir / f"{plot_name}.png"
        fig.savefig(filepath)

    @classmethod
    def _column_to_label_translator(cls, text: str):
        num_id = text[text.rfind("_")+1:]
        if text.startswith("M_kg"):
            return f"Mass {num_id} [-]"
        if text.startswith("K_Nm"):
            return f"K{num_id}/K1 [-]"
        if text.startswith("C_Ns"):
            return f"C{num_id}/C1 [-]"
        if text.startswith("RMS_KE"):
            return f"Kinetic Energy {num_id} [J]"
    param_prefixes = ("K_Nm", "C_Nsm") 

    @classmethod
    def generate_permutation_plots(cls, dof, data_name: str) -> None:
        data = cls.open_data(data_name)


        spring_norm_factor = data.iloc[0]['K_Nm_1']
        damp_norm_factor = data.iloc[0]['C_Nsm_1']
        dimensional_data_slice = data.iloc[0].to_dict()
        # Normalise columns
        norm_k_cols = [c for c in data.columns if c.startswith("K_Nm")]
        norm_c_cols = [c for c in data.columns if c.startswith("C_Nsm")]
        data[norm_k_cols] = data[norm_k_cols] / spring_norm_factor
        data[norm_c_cols] = data[norm_c_cols] / damp_norm_factor
        
        fixed_data_slice = data.iloc[0].to_dict()

        param_prefixes = ("K_Nm", "C_Nsm")
        new_columns = [c for c in data.columns if c.startswith(param_prefixes) and not str(c).endswith("_1")]
        column_permutations = combinations(new_columns, 2)



        fixed_param_prefixes = ("M_kg", "K_Nm", "C_Nsm")
        z = f"RMS_KE_J_{dof}"

        for x, y in column_permutations:
            fixed_vals = {k: v for k, v in fixed_data_slice.items()
                        if k.startswith(fixed_param_prefixes) and k not in (x, y)} 
            dim_fixed_vals = {k: v for k, v in dimensional_data_slice.items()
                                    if k.startswith(fixed_param_prefixes) and k not in (x, y)} 
            
            attr = PlotAttributes(data_name=data_name,
                    plot_name=f"2D_Contour_X{x}_Y{y}_Z{z}", 
                    x_axis=x, 
                    y_axis=y, 
                    z_axis=z, 
                    plot_title=[f"{key}: {value}" for key,value in fixed_vals.items()], 
                    xlabel=cls._column_to_label_translator(x), 
                    ylabel=cls._column_to_label_translator(y), 
                    contour_label=cls._column_to_label_translator(z), 
                    fixed_vals=fixed_vals,
                    logarithmic_scale=True)
            DataPlotter(data).plot_data(attr)

    @classmethod
    def data_row_to_MSD_model(cls, dof, data_row):
        masses, springs, dampers, forces, system = cls._row_to_MSD(dof, data_row)
        model = MassSpringDamperSystem(system)
        return model