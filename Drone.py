import numpy as np

from RPY2Rot import RPY2Rot

class Drone:
    def __init__(self, params, init_state, init_input, gains, sim_time):
        self.g = 9.81
        self.t = 0.0
        self.dt = 0.01
        self.tf = sim_time

        self.m = params['mass']
        self.l = params['arm_length']
        self.I = np.array([
            [params['Ixx'], 0.0, 0.0],
            [0.0, params['Iyy'], 0.0],
            [0.0, 0.0, params['Izz']]
        ])

        self.x = init_state
        self.r = self.x[0:3]
        self.dr = self.x[3:6]
        self.euler = self.x[6:9]
        self.w = self.x[9:12] 

        self.dx = np.zeros((12,1))

        self.u = init_input
        self.F = self.u[0]
        self.M = self.u[1:4]

        self.phi_des = 0.0
        self.phi_err = 0.0
        self.phi_err_prev = 0.0
        self.phi_err_sum = 0.0

        self.theta_des = 0.0
        self.theta_err = 0.0
        self.theta_err_prev = 0.0
        self.theta_err_sum = 0.0

        self.psi_des = 0.0
        self.psi_err = 0.0
        self.psi_err_prev = 0.0
        self.psi_err_sum = 0.0

        self.zdot_des = 0.0
        self.zdot_err = 0.0
        self.zdot_err_prev = 0.0
        self.zdot_err_sum = 0.0

        self.kP_phi = gains['P_phi']
        self.kI_phi = gains['I_phi']
        self.kD_phi = gains['D_phi']

        self.kP_theta = gains['P_theta']
        self.kI_theta = gains['I_theta']
        self.kD_theta = gains['D_theta']

        self.kP_psi = gains['P_psi']
        self.kI_psi = gains['I_psi']
        self.kD_psi = gains['D_psi']

        self.kP_zdot = gains['P_zdot']
        self.kI_zdot = gains['I_zdot']
        self.kD_zdot = gains['D_zdot']

    def eval_EOM(self):
        bRi = RPY2Rot(self.euler)
        R = bRi.T
        
        self.dx[0:3] = self.dr
        self.dx[3:6] = (1 / self.m) * (np.array([[0.0], [0.0], [self.m * self.g]]) + R @ np.array([[0.0], [0.0], [-self.F.item()]]))

        phi = self.euler[0].item()
        theta = self.euler[1].item()

        self.dx[6:9] = np.array([
            [1.0, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0.0, np.cos(phi), -np.sin(phi)],
            [0.0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]
        ]) @ self.w
        
        self.dx[9:12] = np.linalg.inv(self.I) @ (self.M - np.cross(self.w, self.I @ self.w, axis=0))

    def update_state(self):
        self.t += self.dt

        self.eval_EOM()
        self.x = self.x + self.dx * self.dt

        self.r = self.x[0:3]
        self.dr = self.x[3:6]
        self.euler = self.x[6:9]
        self.w = self.x[9:12]
    
    def controller(self, des_state):
        # self.u[0] = self.m * self.g
        # self.u[1] = 0.0
        # self.u[2] = 0.0
        # self.u[3] = 0.0

        self.phi_des = des_state[0]
        self.theta_des = des_state[1]
        self.psi_des = des_state[2]
        self.zdot_des = des_state[3]

        self.phi_err = self.phi_des - self.euler[0].item()
        self.theta_err = self.theta_des - self.euler[1].item()
        self.psi_err = self.psi_des - self.euler[2].item()
        
        self.u[1] = self.kP_phi * self.phi_err + self.kI_phi * self.phi_err_sum + self.kD_phi * (self.phi_err - self.phi_err_prev) / self.dt
        self.phi_err_sum += self.phi_err * self.dt
        self.phi_err_prev = self.phi_err

        self.u[2] = self.kP_theta * self.theta_err + self.kI_theta * self.theta_err_sum + self.kD_theta * (self.theta_err - self.theta_err_prev) / self.dt
        self.theta_err_sum += self.theta_err * self.dt
        self.theta_err_prev = self.theta_err

        self.u[3] = self.kP_psi * self.psi_err + self.kI_psi * self.psi_err_sum + self.kD_psi * (self.psi_err - self.psi_err_prev) / self.dt
        self.psi_err_sum += self.psi_err * self.dt
        self.psi_err_prev = self.psi_err

        self.u[0] = self.m * self.g #  + self.kP_zdot * self.zdot_err + self.kI_zdot * self.zdot_err_sum + self.kD_zdot * (self.zdot_err - self.zdot_err_prev) / self.dt

        self.F = self.u[0]
        self.M = self.u[1:4]

    @property
    def state(self):
        return self.x

