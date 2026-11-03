import numpy as np
import matplotlib.pyplot as plt
from models.AnalysisTools import PlotAttributes, DataPlotter, DataAnalyser

# NAME =  "2DOF_FIXK_1"
# DOF = 2

# df = DataAnalyser.open_data(NAME)
# df['Para'] = (df['RMS_KE_J_2'] - df['RMS_KE_J_1'])
# row = DataAnalyser.locate_data_maxima(df, ['Para'])
# DataAnalyser.validate_data_row(DOF, row)
# # Dataset.dataset_to_excel(NAME, df)
# # PARAMETERS = ["M_kg", "C_Nsm", "K_Nm", "RMS_KE_J"]
# attr = PlotAttributes(NAME,'2D_Contour','C_Nsm_2','RMS_KE_J_2',None, None, 'C_2', 'KE_2', None)
# plotter = DataPlotter(DOF, df)
# plotter.plot_data(attr)

# NAME =  "2DOF_1"
# DOF = 2
# df = DataAnalyser.open_data(NAME)

# df['KE_diff'] = (df['RMS_KE_J_2'] - df['RMS_KE_J_1'])
# print(df)
# attr = PlotAttributes(data_name=NAME,
#                       plot_name="2D_contourplot",
#                       x_axis='K_Nm_2',
#                       y_axis='C_Nsm_2',
#                       z_axis='KE_diff', 
#                       plot_title=None,
#                       xlabel='K_2', 
#                       ylabel='C_2',
#                       contour_label='KE2-KE1')
# plotter = DataPlotter(DOF, df)
# plotter.plot_data(attr)

NAME =  "3DOF_1"
DOF = 3
df = DataAnalyser.open_data(NAME, columns=['RMS_KE_J_1','RMS_KE_J_3','K_Nm_3','C_Nsm_3'])
print("a")
df['KE_diff'] = (df['RMS_KE_J_3'] - df['RMS_KE_J_1'])
attr = PlotAttributes(data_name=NAME,
                      plot_name="2D_contourplot",
                      x_axis='K_Nm_3',
                      y_axis='C_Nsm_3',
                      z_axis='KE_diff', 
                      plot_title=None,
                      xlabel='K_3', 
                      ylabel='C_3',
                      contour_label='KE3-KE1')
plotter = DataPlotter(DOF, df)
print("a")
plotter.plot_data(attr)