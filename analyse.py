import numpy as np
import matplotlib.pyplot as plt
from models.AnalysisTools import PlotAttributes, DataPlotter, DataAnalyser

NAME =  "3DOF_TEST_1_FIX_C"
DOF = 3

df = DataAnalyser.open_data(NAME)
df['Para'] = (df['RMS_KE_J_2'] - df['RMS_KE_J_1'])
row = DataAnalyser.locate_data_maxima(df, ['Para'])
DataAnalyser.validate_data_row(DOF, row)
# Dataset.dataset_to_excel(NAME, df)
# PARAMETERS = ["M_kg", "C_Nsm", "K_Nm", "RMS_KE_J"]
attr = PlotAttributes(NAME,'2D_Contour','K_Nm_2','RMS_KE_J_2',None, None, 'K_2', 'KE_2', None)
plotter = DataPlotter(DOF, df)
plotter.plot_data(attr)
plt.show()