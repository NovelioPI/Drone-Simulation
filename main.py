import numpy as np
import matplotlib.pyplot as plt
import time

from Drone import Drone
from RPY2Rot import RPY2Rot

plt.ion()

# DEFINE
R2D = 180.0 / np.pi # rad to deg
D2R = np.pi / 180.0 # deg to rad

# INIT PARAMS
drone_params = {
    'mass': 1.25, # kg
    'arm_length': 0.265, # m
    'Ixx': 0.0232, # kg*m^2
    'Iyy': 0.0232, # kg*m^2
    'Izz': 0.0468, # kg*m^2
}

drone_init_state = np.array([[
    0.0, 0.0, -5.0,								# X, Y, Z
    0.0, 0.0, 0.0,								# dX, dY, dZ
    0.0, 0.0, 0.0,								# phi, theta, psi
    0.0, 0.0, 0.0,								# p, q, r
]]).T

drone_init_input = np.array([[0.0, 0.0, 0.0, 0.0]]).T   # u1, u2, u3, u4 (F, M1, M2, M3)

drone_body = np.array([
    [ 0.265,    0.0,   0.0, 1.0],
    [   0.0, -0.265,   0.0, 1.0],
    [-0.265,    0.0,   0.0, 1.0],
	[   0.0,  0.265,   0.0, 1.0],
	[   0.0,    0.0,   0.0, 1.0],
	[   0.0,    0.0, -0.15, 1.0]
]).T

# change the gains here, tuning is required
drone_gains = {
    'P_phi': 0.0, 'I_phi': 0.0, 'D_phi': 0.0,
    'P_theta': 0.0, 'I_theta': 0.0, 'D_theta': 0.0,
    'P_psi': 0.0, 'I_psi': 0.0, 'D_psi': 0.0,
    'P_zdot': 0.0, 'I_zdot': 0.0, 'D_zdot': 0.0
}

sim_time = 2.0

drone = Drone(drone_params, drone_init_state, drone_init_input, drone_gains, sim_time)

# INIT 3D PLOT
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_xlim(-5.0, 5.0)
ax.set_ylim(-5.0, 5.0)
ax.set_zlim(-8.0, 0.0)
ax.invert_yaxis()
ax.invert_zaxis()

drone_state = drone.state
wHb = np.vstack((np.hstack((RPY2Rot(drone_state[6:9]).T, drone_state[0:3])), np.array([[0.0, 0.0, 0.0, 1.0]])))
drone_world = wHb @ drone_body
drone_attitude = drone_world[0:3, :]

fig_arm13 = ax.plot(drone_attitude[0, [0, 2]], drone_attitude[1, [0, 2]], drone_attitude[2, [0, 2]], '-ro', markersize=5)
fig_arm24 = ax.plot(drone_attitude[0, [1, 3]], drone_attitude[1, [1, 3]], drone_attitude[2, [1, 3]], '-bo', markersize=5)
fig_payload = ax.plot(drone_attitude[0, [4, 5]], drone_attitude[1, [4, 5]], drone_attitude[2, [4, 5]], '-k', linewidth=3.0)
fig_shadow = ax.plot(0, 0, 0, 'xk', linewidth=3.0)

# INIT DATA PLOT
fig2 = plt.figure()
ax2 = fig2.subplots(3, 2)
ax2[0, 0].set_title('phi [deg]')
ax2[0, 1].set_title('theta [deg]')
ax2[1, 0].set_title('psi [deg]')
ax2[1, 1].set_title('X [m]')
ax2[2, 0].set_title('Y [m]')
ax2[2, 1].set_title('dZ [m/s]')

# SIMULATION
# change the desired state here
des_state = np.array([[
    0.0,    # roll      example: 10.0 * D2R
    0.0,    # pitch
    0.0,    # yaw
    0.0     # zdot
]]).T

timearray = np.arange(0.0, sim_time, 0.01)
for i in timearray:
    drone.controller(des_state)
    drone.update_state()

    drone_state = drone.state

    # UPDATE 3D PLOT
    wHb = np.vstack((np.hstack((RPY2Rot(drone_state[6:9]).T, drone_state[0:3])), np.array([[0.0, 0.0, 0.0, 1.0]])))
    drone_world = wHb @ drone_body
    drone_attitude = drone_world[0:3, :]

    fig_arm13[0].set_data(drone_attitude[0, [0, 2]], drone_attitude[1, [0, 2]])
    fig_arm13[0].set_3d_properties(drone_attitude[2, [0, 2]])
    fig_arm24[0].set_data(drone_attitude[0, [1, 3]], drone_attitude[1, [1, 3]])
    fig_arm24[0].set_3d_properties(drone_attitude[2, [1, 3]])
    fig_payload[0].set_data(drone_attitude[0, [4, 5]], drone_attitude[1, [4, 5]])
    fig_payload[0].set_3d_properties(drone_attitude[2, [4, 5]])
    fig_shadow[0].set_data(drone_attitude[0, 4], drone_attitude[1, 4])
    fig_shadow[0].set_3d_properties(0)

    fig.canvas.draw()
    fig.canvas.flush_events()

    # UPDATE DATA PLOT
    ax2[0, 0].plot(i, drone_state[6] * R2D, 'r.')
    ax2[0, 1].plot(i, drone_state[7] * R2D, 'r.')
    ax2[1, 0].plot(i, drone_state[8] * R2D, 'r.')
    ax2[1, 1].plot(i, drone_state[0], 'r.')
    ax2[2, 0].plot(i, drone_state[1], 'r.')
    ax2[2, 1].plot(i, drone_state[5], 'r.')
    fig2.canvas.draw()
    fig2.canvas.flush_events()

    if drone_state[2] >= 0:
        break