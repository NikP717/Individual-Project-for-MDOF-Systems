import numpy as np
import matplotlib.pyplot as plt
from models.AnalysisTools import PlotAttributes, DataPlotter, DataAnalyser

# NAME =  "2DOF_FIXK_1"
# DOF = 2
names = ["2DOF_S1", "3DOF_S1", "4DOF_S1"]
# DataAnalyser.generate_permutation_plots(3, NAME)
for name in names:
    data = DataAnalyser.open_data(name)
    DataAnalyser.generate_receptance_plot(data, name)
    DataAnalyser.generate_permutation_plots(data, name)
    # DataAnalyser.generate_permutation_plots_old(data, name)
