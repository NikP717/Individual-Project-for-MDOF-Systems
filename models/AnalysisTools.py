import pandas as pd
from dataclasses import dataclass
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from models.SimulationData import Dataset
from models.DOFModel import MassSpringDamper, MassSpringDamperSystem
from scipy.integrate import trapezoid
from itertools import  combinations
import re
import textwrap
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

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
    def generate_receptance_plot(cls, data, data_name, masses: list[int] | None = None) -> None:
        dof = max(int(m.group(1)) for col in data.columns if (m := re.fullmatch(r"X_real_m_(\d+)", col)))
        force = 1  # NOTE: HARD CODED VALUE
        spring_norm_factor = data.iloc[0]["K_Nm_1"]
        damp_norm_factor = data.iloc[0]["C_Nsm_1"]
        # Normalise columns
        norm_k_cols = [c for c in data.columns if c.startswith("K_Nm")]
        norm_c_cols = [c for c in data.columns if c.startswith("C_Nsm")]
        data[norm_k_cols] = data[norm_k_cols] / spring_norm_factor
        data[norm_c_cols] = data[norm_c_cols] / damp_norm_factor
        param_cols = [col for col in data.columns if col.startswith(("M_kg_", "C_Nsm_", "K_Nm_"))]
        # Default: every mass except mass 1 (2DOF -> [2], 3DOF -> [2, 3])
        if masses is None:
            masses = list(range(2, dof + 1))
        # Per-mass data: vary only that mass's K and C, lock every other parameter to its first-row value
        sub, limits = {}, {}
        for mass in masses:
            varied = (f"C_Nsm_{mass}", f"K_Nm_{mass}")
            mask = np.ones(len(data), dtype=bool)
            for col in param_cols:
                if col not in varied:
                    mask &= np.isclose(data[col], data[col].iloc[0])
            d = data[mask].copy()
            d["R_real"], d["R_imag"] = d[f"X_real_m_{mass}"] / force, d[f"X_imag_m_{mass}"] / force
            sub[mass] = d
            # Square limits centred on this mass's data (5% margin each side)
            re_vals, im_vals = d["R_real"].to_numpy(), d["R_imag"].to_numpy()
            re_vals, im_vals = re_vals[np.isfinite(re_vals)], im_vals[np.isfinite(im_vals)]
            half = max(np.ptp(re_vals), np.ptp(im_vals), 1e-12) * 0.55
            x_mid, y_mid = (re_vals.max() + re_vals.min()) / 2, (im_vals.max() + im_vals.min()) / 2
            limits[mass] = ((x_mid - half, x_mid + half), (y_mid - half, y_mid + half))
        fig, axes = plt.subplots(1, len(masses), figsize=(6.8 * len(masses), 6.8), squeeze=False, layout="constrained")
        axes = axes[0]
        c_colour, k_colour = "#1f4e79", "#d1495b"
        for ax, mass in zip(axes, masses):
            c_col, k_col = f"C_Nsm_{mass}", f"K_Nm_{mass}"
            plot_df = sub[mass]
            x_lim, y_lim = limits[mass]
            x_span, y_span = np.ptp(x_lim), np.ptp(y_lim)
            for fam, vary, sty, col, lab in ((c_col, k_col, "-", c_colour, "K"), (k_col, c_col, "--", k_colour, "C")):
                lines = []
                for _, g in plot_df.groupby(fam):
                    g = g.sort_values(vary)
                    x, y = g["R_real"].to_numpy(dtype=float), g["R_imag"].to_numpy(dtype=float)
                    ax.plot(x, y, sty, lw=1.0, color=col, alpha=0.8)
                    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x) / x_span, np.diff(y) / y_span))])
                    lines.append((s[-1], s, x, y))
                # Direction arrow on the longest line, spanning 40%-60% of its length
                total, s, x, y = max(lines, key=lambda t: t[0])
                if total <= 0:
                    continue
                s = s / total
                (x0, x1), (y0, y1) = np.interp([0.4, 0.6], s, x), np.interp([0.4, 0.6], s, y)
                arrow = ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=5,
                                    arrowprops=dict(arrowstyle="-|>", color="black", lw=2.5, mutation_scale=22, shrinkA=0, shrinkB=0))
                arrow.arrow_patch.set_path_effects([pe.withStroke(linewidth=4.5, foreground="white")])
                ax.annotate(rf"{lab}$_{{{mass}}}$/{lab}$_1$ increasing", xy=((x0 + x1) / 2, (y0 + y1) / 2), xytext=(0, 14),
                            textcoords="offset points", ha="center", fontsize=9, fontweight="bold", zorder=6,
                            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.6, alpha=0.95))
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
            if x_lim[0] < 0 < x_lim[1]:
                ax.axvline(0, color="grey", lw=0.6, zorder=0)
            if y_lim[0] < 0 < y_lim[1]:
                ax.axhline(0, color="grey", lw=0.6, zorder=0)
            ax.set_aspect("equal", adjustable="box")
            ax.set_box_aspect(1)
            ax.grid(True, alpha=0.25, linestyle=":")
            ax.ticklabel_format(style="sci", scilimits=(-3, 3))
            ax.set_xlabel(rf"Re($\alpha_{{{mass}1}}$)  [m/N]")
            ax.set_ylabel(rf"Im($\alpha_{{{mass}1}}$)  [m/N]")
            ax.set_title(f"Mass {mass}", fontweight="bold")
            # Per-panel legend, since each mass varies different parameters
            handles = [
                Line2D([], [], ls="-", color=c_colour, lw=1.5, label=f"Constant {cls._column_to_label_translator(c_col)}"),
                Line2D([], [], ls="--", color=k_colour, lw=1.5, label=f"Constant {cls._column_to_label_translator(k_col)}"),
            ]
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
        title = f"Receptance for {dof}DOF System: F1 = 1N, M1&2 = 1kg, K1 = 100N/m, C1 = 1 Ns/m"
        wrap_width = max(30, int(fig.get_figwidth() * 9))
        fig.suptitle("\n".join(textwrap.wrap(title, width=wrap_width)), fontsize=13, fontweight="bold")
        mass_tag = "_".join(str(m) for m in masses)
        plot_name = f"Receptance Plot with varied K and C of masses {mass_tag}"
        out_dir = Path(__file__).resolve().parent.parent / "DATA" / data_name
        fig.savefig(out_dir / f"{plot_name}.png", dpi=200)
        plt.close(fig)

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
    def generate_permutation_plots(cls, data, data_name) -> None:
        data = data.copy()
        dof = max(int(m.group(1)) for col in data.columns if (m := re.fullmatch(r"X_real_m_(\d+)", col)))
        spring_norm_factor = data.iloc[0]["K_Nm_1"]
        damp_norm_factor = data.iloc[0]["C_Nsm_1"]
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
        out_dir = Path(__file__).resolve().parent.parent / "DATA" / data_name
        # Iso-lines as fractions of each slice's own peak: labelled major lines plus unlabelled minor lines, packed towards the peak
        major_fracs = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        minor_fracs = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.85, 0.925, 0.96, 0.97, 0.98, 0.985, 0.995, 0.998]
        line_colour = "black"
        minor_lw, minor_alpha = 0.7, 0.5
        major_lw, major_alpha = 0.9, 0.8

        def use_log(v):
            # Log axis whenever the sweep is positive and spans a decade or more
            v = np.asarray(v, dtype=float)
            return v.min() > 0 and v.max() / v.min() >= 10

        # Pass 1: build every slice so one colour scale can span the whole parameter space
        slices = []
        for x, y in column_permutations:
            fixed_vals = {k: v for k, v in fixed_data_slice.items()
                          if k.startswith(fixed_param_prefixes) and k not in (x, y)}
            mask = np.ones(len(data), dtype=bool)
            for k, v in fixed_vals.items():
                mask &= np.isclose(data[k], v)
            grid = data[mask].pivot_table(index=y, columns=x, values=z, aggfunc="max")
            if grid.shape[0] < 2 or grid.shape[1] < 2:
                continue
            slices.append((x, y, fixed_vals, grid))
        if not slices:
            return
        all_z = np.concatenate([g.to_numpy(dtype=float).ravel() for *_, g in slices])
        all_z = all_z[np.isfinite(all_z)]
        z_min, z_max = all_z.min(), all_z.max()
        # One linear scale shared by every figure, so a colour means the same value everywhere
        levels = np.linspace(z_min, z_max, 256)
        ticks = np.linspace(z_min, z_max, 10)
        # Pass 2: plot
        for x, y, fixed_vals, grid in slices:
            X, Y = np.meshgrid(grid.columns.to_numpy(dtype=float), grid.index.to_numpy(dtype=float))
            Z = grid.to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
            cf = ax.contourf(X, Y, Z, levels=levels, cmap="rainbow", vmin=z_min, vmax=z_max)
            cbar = fig.colorbar(cf, ax=ax, pad=0.02, ticks=ticks)
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2e"))
            cbar.set_label(cls._column_to_label_translator(z))
            z_peak = np.nanmax(Z)
            z_lo = np.nanmin(Z)
            minor = [f for f in minor_fracs if f * z_peak > z_lo]
            major = [f for f in major_fracs if f * z_peak > z_lo]
            if minor:
                ax.contour(X, Y, Z, levels=[f * z_peak for f in minor], colors=line_colour, linewidths=minor_lw, alpha=minor_alpha)
            if major:
                cs = ax.contour(X, Y, Z, levels=[f * z_peak for f in major], colors=line_colour, linewidths=major_lw, alpha=major_alpha)
                ax.clabel(cs, fmt={f * z_peak: f"{f:.1%}".replace(".0%", "%") for f in major},
                          fontsize=7, colors=line_colour, inline=True, inline_spacing=6)
            # Legend entries (proxy artists, so the legend matches the line styles exactly)
            handles = [
                Line2D([], [], color=line_colour, lw=major_lw, alpha=major_alpha, label="Labelled contours (% of max on this plot)"),
                Line2D([], [], color=line_colour, lw=minor_lw, alpha=minor_alpha, label="Intermediate contours"),
            ]
            ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8)
            if use_log(X[0]):
                ax.set_xscale("log")
            if use_log(Y[:, 0]):
                ax.set_yscale("log")
            ax.set_xlim(X.min(), X.max())
            ax.set_ylim(Y.min(), Y.max())
            ax.grid(True, alpha=0.2, linestyle=":", which="both")
            ax.set_xlabel(cls._column_to_label_translator(x))
            ax.set_ylabel(cls._column_to_label_translator(y))
            # Group equal masses, e.g. {1.0: [1, 2]} -> "M1&2 = 1kg"
            mass_groups = {}
            for k, v in fixed_vals.items():
                if k.startswith("M_kg"):
                    mass_groups.setdefault(v, []).append(k.split("_")[-1])
            mass_txt = ", ".join(f"M{'&'.join(ids)} = {v:.3g}kg" for v, ids in mass_groups.items())
            # Fixed ratios of the other springs and dampers (K_n/K_1, C_n/C_1); the 1 entries are the dimensional base values
            ratio_txt = ", ".join(
                f"{k.split('_')[0]}{k.split('_')[-1]}/{k.split('_')[0]}1 = {v:.3g}"
                for k, v in fixed_vals.items()
                if k.startswith(("K_Nm", "C_Nsm")) and not k.endswith("_1")
            )
            base_txt = f"K1 = {spring_norm_factor:.3g} N/m, C1 = {damp_norm_factor:.3g} Ns/m"
            line1 = f"{cls._column_to_label_translator(z)} for {dof}DOF System: {mass_txt}, {base_txt}"
            line2 = f"Swept: {cls._column_to_label_translator(x)} vs {cls._column_to_label_translator(y)}"
            if ratio_txt:
                line2 += f"   |   Held at: {ratio_txt}"
            wrap_width = max(40, int(fig.get_figwidth() * 11))
            fig.suptitle("\n".join(textwrap.wrap(line1, wrap_width) + textwrap.wrap(line2, wrap_width)),
                         fontsize=12, fontweight="bold")
            
            plot_name = f"2D_Contour_X{x}_Y{y}_Z{z}"
            fig.savefig(out_dir / f"{plot_name}.png", dpi=200)
            plt.close(fig)
            
    @classmethod
    def generate_permutation_plots_old(cls, data, data_name) -> None:
        dof = max(int(m.group(1)) for col in data.columns if (m := re.fullmatch(r"X_real_m_(\d+)", col)))
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
                    plot_name=f"2D_Contour_X{x}_Y{y}_Z{z}_old", 
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