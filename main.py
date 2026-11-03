from models.ForwardDesignSearchMethods import ManualParameterSweep, MethodConfig
from dataclasses import dataclass
WALL_CONNECT_SPRING = 100 #N/m
WALL_FORCE = 1 #N
WALL_CONNECT_DAMPER = 1 #Ns/m
WALL_MASS = 1 # kg

@dataclass
class MethodAttr:
    name: str
    dof: int
    k_range: list
    k_step: int
    c_range: list
    c_step: int
    fix_k: bool = False
    fix_c: bool = False

    def unwrap_attr(self):
        return [self.name, self.dof, self.k_range, self.k_step, self.c_range, self.c_step, self.fix_k, self.fix_c]

def run_config_and_method(name, dof, k_range, k_step, c_range, c_step, fix_k=False, fix_c=False):
    config = MethodConfig(dataset_name=name, DOF=dof, wall_connection_mass=WALL_MASS, wall_connection_spring=WALL_CONNECT_SPRING,
                        wall_connection_damper=WALL_CONNECT_DAMPER,wall_force=WALL_FORCE,k_range=k_range,kstep=k_step,c_range=c_range,cstep=c_step)
    method = ManualParameterSweep(config)
    method.run(fix_k=fix_k, fix_c=fix_c)


def main():
    # method = MethodAttr(name="2DOF_FIXC_1",dof=2,k_range=[1,100000],k_step=0.1,c_range=None,c_step=None,fix_k=False,fix_c=True)
    # run_config_and_method(*method.unwrap_attr())
    # method = MethodAttr(name="2DOF_FIXK_1",dof=2,k_range=None,k_step=None,c_range=[0.01,5],c_step=0.001,fix_k=True,fix_c=False)
    # run_config_and_method(*method.unwrap_attr())
    method = MethodAttr(name='2DOF_2', dof = 2, k_range = [1, 1000], k_step=1, c_range = [0.01, 10], c_step=0.01, fix_k=False,fix_c=False)
    run_config_and_method(*method.unwrap_attr())
    # method = MethodAttr(name='3DOF_1', dof = 3, k_range = [1, 10000], k_step=100, c_range = [0.01, 5], c_step=0.1, fix_k=False,fix_c=False)
    # run_config_and_method(*method.unwrap_attr())
    # do large data sweeps for 2,3,4 DOF systems repeatedly with new plotting systems and see what data we can draw
    # Once done add an optimisiation inverse design stream by adding anew code 

if __name__ == "__main__":
    main()
    # max_row_collection = df.sort_values("RMS_KE_J_3",ascending=False).head(100)
