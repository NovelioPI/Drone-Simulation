import numpy as np

def RPY2Rot(euler):
    phi = euler[0].item()
    theta = euler[1].item()
    psi = euler[2].item()
    
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(phi), np.sin(phi)],
        [0, -np.sin(phi), np.cos(phi)]
    ])

    R_y = np.array([
        [np.cos(theta), 0, -np.sin(theta)],
        [0, 1, 0],
        [np.sin(theta), 0.0, np.cos(theta)]
    ])

    R_z = np.array([
        [np.cos(psi), np.sin(psi), 0],
        [-np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])

    bRi = R_x @ R_y @ R_z

    return bRi