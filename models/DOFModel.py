import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp

@dataclass
class MassSpringDamper():
    DOF: int # 1,2,3,4... N masses
    masses: list # masses in kg
    springs: list # list of spring constants N/m
    dampers: list # list of dampers
    forces: list # list of forces applied to each mass
    frequency_hz: float = None # Hz

    def check_MSD(self) -> None:
        con1 = len(self.masses) != self.DOF
        con2 = len(self.springs) != self.DOF
        con3 = len(self.dampers) != self.DOF
        con4 = len(self.forces) != self.DOF

        if con1 or con2 or con3 or con4:
            raise ValueError("Number of elements inputted is not the same as required degree of freedoms.")


class MassSpringDamperSystem():
    def __init__(self, mass_spring_damper: MassSpringDamper) -> None:
        self.MSD = mass_spring_damper
        self.MSD.check_MSD()
        self.dof = self.MSD.DOF

        self.empty_row = [0 for i in range(self.dof)]
        self.empty_matrix = np.array([self.empty_row for i in range(self.dof)])
    
        self._set_masses(self.MSD.masses)
        self._set_damping(self.MSD.dampers)
        self._set_springs(self.MSD.springs)
        self._set_forces(self.MSD.forces)
        if self.MSD.frequency_hz:
            self.fq_HZ = self.MSD.frequency_hz
        else:
            self.fq_HZ = self._calculate_undamped_natural_fq()
        # print(self.fq_HZ)

        # solution matrices
        self.impedance_matrix = None
        self.velocity_matrix = None

        # solution mass array
        self.mass_states = []
        self.KE = [] # or samplerow shortcut

    def calculate_SS_system(self, display = False):
        self._calculate_impedance_matrix()
        self._calculate_velocity_matrix()
        # self._set_mass_states()
        # if display:
        #     self._display_results()
        velocity_complex = self.velocity_matrix
        velocity_peaks = np.abs(velocity_complex)
        rms_KE = 1/4 * np.array(self.MSD.masses) * velocity_peaks**2
        self.KE = rms_KE.tolist()

#           def peak_velocity(self):
# #         return abs(self.velocity_complex)

#     @property
#     def rms_kinetic_energy(self):
#         return (1/4 * self.mass * self.peak_velocity**2)
    #         current_mass = MassProperty(
    #             number = i+1,
    #             mass = self.MSD.masses[i],
    #             velocity_complex = self.velocity_matrix[i,0],
    #             forced_fq = self.fq_HZ
    #         )
    #         self.mass_states.append(current_mass)
    #         KE.append(float(current_mass.rms_kinetic_energy))
    #     self.KE = KE

    def calculate_dynamic_system(self, t_range):
        """
        Uses state space equations: (q1,q2,q3...q1_dot,q2_dot,q3_dot...) as state vector, look at paper for derivation will need to put in lit review
        Ouputs: Array of all masses velocities and displacements with time as a dynamic solution ([x1(t1),x1(t2),...], [x2(t1),x2(t2),...], ...)
        """
        # q = x1, q_dot = x2 (state space to hold x1 displacement vector of all masses, x2 to hold velocity vector of all masses)
        # 0 initial conditions, 0 velocity and displacement
        N = self.dof
        q0 = np.zeros(N)
        q0_dot = np.zeros(N)
        x0 = np.concatenate((q0, q0_dot)) # state space vector
        fq = self.fq_HZ
        dt = 0.01

        F = self.force_matrix
        M = self.mass_matrix
        K = self.spring_matrix
        C = self.damping_matrix
        M_inv = np.linalg.inv(M)

        t_eval = np.arange(t_range[0],t_range[1] + dt, dt)

        def _dynamic_force(t): 
            return F[0]*np.cos(fq*2*np.pi*t)

        def _ndof_system_state_space(t, x):
            q = x[:N]
            q_dot = x[N:]
            F_current = F.copy()
            F_current[0] = _dynamic_force(t)
            q_ddot = (M_inv @ (F_current - C @ q_dot - K @ q)).flatten()
            return np.concatenate([q_dot, q_ddot])
        
        sol = solve_ivp(_ndof_system_state_space,t_range,x0,t_eval=t_eval, method='RK45')
        displacement_array = sol.y[:N, :]
        velocity_array = sol.y[N:, :]
        time = sol.t
        return displacement_array, velocity_array, time
        
    def _set_masses(self, mass_array: list) -> None:
        """mass_array starting from m1 to mN"""
        mass_matrix = np.diag(mass_array)
        mass_matrix = np.array(mass_matrix)
        self.mass_matrix = mass_matrix

    def _assemble_connection_matrix(self, values):
        matrix = np.zeros((self.dof, self.dof), dtype=float)
        for i, value in enumerate(values):
            matrix[i, i] += value
            if i > 0: # ENSURING not out of bounds condition occurs
                matrix[i - 1, i - 1] += value
                matrix[i, i - 1] -= value
                matrix[i - 1, i] -= value
        return matrix

    def _set_springs(self, spring_array: list) -> None:
        self.spring_matrix = np.array(self._assemble_connection_matrix(spring_array))

    def _set_damping(self, damping_array: list) -> None:
        self.damping_matrix = np.array(self._assemble_connection_matrix(damping_array))

    def _set_forces(self, force_array: list) -> None:
        self.force_matrix = np.array(force_array)

    def _calculate_undamped_natural_fq(self):
        """Calculates uncoupled frequency of first mass as connection stays constant"""
        return np.sqrt(float(self.MSD.springs[0])/float(self.MSD.masses[0])) / (2*np.pi)

    def _calculate_impedance_matrix(self):
        fq_RAD = self.fq_HZ *2*np.pi
        s = 1j * fq_RAD
        M = self.mass_matrix
        K = self.spring_matrix
        C = self.damping_matrix
        Z = M*s + K * 1/s + C
        self.impedance_matrix = Z

    def _calculate_velocity_matrix(self):
        Z = self.impedance_matrix
        F = self.force_matrix
        V = np.linalg.solve(Z, F)
        self.velocity_matrix = V           



# MSDS = MassSpringDamperSystem(MassSpringDamper(DOF=3,masses=[1,1,1],springs=[500,100,200],dampers=[0.5,0.1,0.1],forces=[1,0,0]))
# displacement, velocity, time = MSDS.calculate_dynamic_system([100,120])

# from matplotlib import pyplot as plt
# plt.plot(time, displacement[0, :]*1000,label="Mass 1")
# plt.plot(time, displacement[1, :]*1000,label="Mass 2")
# plt.plot(time, displacement[2, :]*1000,label="Mass 3")
# plt.xlabel("Time (s)")
# plt.ylabel("Displacement (mm)")
# plt.show()


# @dataclass
# class MassProperty():
#     number: int # 1,2,3,4,...N mass
#     mass: float # kg
#     velocity_complex: complex # COMPLEX VELOCITY VALUE
#     forced_fq: float # Hz FORCING FREQUENCY 

#     @property
#     def peak_velocity(self):
#         return abs(self.velocity_complex)

#     @property
#     def rms_kinetic_energy(self):
#         return (1/4 * self.mass * self.peak_velocity**2)

#     def display_state(self):
#         print(
#         f"""---------------------------------
#         Mass {self.number}: {self.mass} kg
#         Average KE: {self.rms_kinetic_energy:.2f} J 
#         Steady State Peak Velocity: {self.peak_velocity:.2f} m/s""")


    # def _set_mass_states(self):
    #     KE = []
    #     for i in range(0, self.dof):
    #         current_mass = MassProperty(
    #             number = i+1,
    #             mass = self.MSD.masses[i],
    #             velocity_complex = self.velocity_matrix[i,0],
    #             forced_fq = self.fq_HZ
    #         )
    #         self.mass_states.append(current_mass)
    #         KE.append(float(current_mass.rms_kinetic_energy))
    #     self.KE = KE

    
    # def _display_results(self):
    #     for mass in self.mass_states:
    #         mass.display_state()