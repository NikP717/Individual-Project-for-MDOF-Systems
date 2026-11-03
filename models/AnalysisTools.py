import pandas as pd
from dataclasses import dataclass
from matplotlib import pyplot as plt
from pathlib import Path
import matplotlib.tri as mtri
import numpy as np
from models.SimulationData import Dataset
from models.DOFModel import MassSpringDamper, MassSpringDamperSystem
from scipy.integrate import trapezoid

@dataclass
class PlotAttributes():
    data_name: str
    plot_name: str
    x_axis: str
    y_axis: str
    z_axis: str
    plot_title: str = None
    xlabel: str = None
    ylabel: str = None
    contour_label: str = None

class DataPlotter():
    def __init__(self, DOF: float, df: pd.DataFrame):
        self.DOF = DOF
        self.df = df

    def plot_data(self, attr: PlotAttributes):
        if hasattr(attr, 'z_axis'):
            if attr.z_axis is not None:
                self._contour_plot(attr)
                return
        self._2d_plot(attr)

    def _contour_plot(self, attr: PlotAttributes):
        df = self.df
        Z = df.pivot(index=attr.y_axis,columns=attr.x_axis,values=attr.z_axis).sort_index().sort_index(axis=1)

        fig, ax = plt.subplots(figsize=(10, 7))
        contour = ax.contourf(Z.columns.to_numpy(),Z.index.to_numpy(),Z.to_numpy(),levels=100,cmap="viridis")
        fig.colorbar(contour, ax=ax, label=attr.contour_label)

        ax.set_xlabel(attr.xlabel)
        ax.set_ylabel(attr.ylabel)
        ax.set_title(attr.plot_name)
        fig.tight_layout()
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

        self._save_plot(fig, attr)

    def _save_plot(self, fig: plt.figure, attr: PlotAttributes):
        dir =  Path(__file__).resolve().parent.parent / "DATA" / attr.data_name
        filepath = dir / f"{attr.plot_name}.png"
        fig.savefig(filepath)

    # plot_smooth_contour(
    #     df,
    #     'K_Nm_3', 'K_Nm_2', 'RMS_KE_J_3',
    #     'Spring stiffness 3 (N/m)',
    #     'Spring stiffness 2 (N/m)',
    #     "Spring_stiffness_design_space"
    # )

    # plot_smooth_contour(
    #     df,
    #     'C_Nsm_3', 'C_Nsm_2', 'RMS_KE_J_3',
    #     'Damping coefficient 3 (Ns/m)',
    #     'Damping coefficient 2 (Ns/m)',
    #     "Damping_design_space"
    # )

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