import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.linalg import eigh


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
            self.fq_HZ = self.calculate_undamped_natural_fq()
        # print(self.fq_HZ)

        # solution matrices
        self.impedance_matrix = None
        self.velocity_matrix = None

        # solution mass array
        self.mass_states = []
        self.KE = [] # for samplerow shortcut
        self.X = [] # for samplerow shortcut

    def calculate_SS_system(self, display = False):
        self._calculate_impedance_matrix()
        self._calculate_velocity_matrix()

        velocity_complex = self.velocity_matrix
        velocity_peaks = np.abs(velocity_complex)
        rms_KE = 1/4 * np.array(self.MSD.masses) * velocity_peaks**2
        self.KE = rms_KE.tolist()
        X = (velocity_complex/(2*self.fq_HZ*np.pi*1j))
        self.X_real = X.real.tolist()
        self.X_imag = X.imag.tolist()

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

    def calculate_undamped_natural_fq(self):
        """Calculates uncoupled frequency of first mass as connection stays constant"""
        return np.sqrt(float(self.MSD.springs[0])/float(self.MSD.masses[0])) / (2*np.pi)

    def calculate_coupled_natural_fq(self):
        """Undamped coupled natural frequencies (Hz) and mode shapes."""
        eigvals, modes = eigh(self.spring_matrix, self.mass_matrix)  # K v = w^2 M v
        omega = np.sqrt(eigvals)          # rad/s, ascending
        return omega / (2 * np.pi)

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