import numpy as np
import matplotlib.pyplot as plt
from models.AnalysisTools import PlotAttributes, DataPlotter, DataAnalyser

# NAME =  "2DOF_FIXK_1"
# DOF = 2

NAME =  "2DOF_3_X"
# DataAnalyser.generate_permutation_plots(3, NAME)
# DataAnalyser.generate_receptance_plot(2, NAME)
# DataAnalyser.generate_permutation_plots(2, NAME)

data = DataAnalyser.open_data(NAME)
row = DataAnalyser.locate_data_maxima(data, 'RMS_KE_J_2')
model = DataAnalyser.data_row_to_MSD_model(2, row)
print(model.calculate_coupled_natural_fq())