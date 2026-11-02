from models.SimulationData import Dataset
import numpy as np
import matplotlib.pyplot as plt
from models.ForwardDesignSearchMethods import ManualParameterSweep, MethodConfig
from models.AnalysisTools import PlotAttributes, DataPlotter, DataAnalyser

def main():
    NAME = "3DOF_TEST_1_FIX_C"
    DOF = 3

    k_range = [1,10000]
    k_step = 1
    c_range = [0.01,5]
    c_step = 0.1
    config = MethodConfig(dataset_name=NAME, DOF=DOF, wall_connection_mass=1, wall_connection_spring=100,
                          wall_connection_damper=1,wall_force=1,k_range=k_range,kstep=k_step,c_range=c_range,cstep=c_step)
    method = ManualParameterSweep(config)
    method.run()
    
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
    #TODO: Create automated data analysis regardless of DOF to get rid of this bottom bottle neck, same with data verification 
    # (make it so i can take any row and plot it dynamically and compare the difference)
    # do large data sweeps for 2,3,4 DOF systems repeatedly with new plotting systems and see what data we can draw
    # Once done add an optimisiation inverse design stream by adding anew code 

if __name__ == "__main__":
    main()
    # max_row_collection = df.sort_values("RMS_KE_J_3",ascending=False).head(100)
