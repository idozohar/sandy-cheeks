import numpy as np
from mujoco_py import (functions, MjSim, MjSimState)


def find_contacts(sim, gripper_geom):
    # TABLE = sim.model.body_name2id('OUR_TABLE')
    # box = sim.model.body_name2id('box')
    gripper_geom_id = sim.model.geom_name2id(gripper_geom)
    wall_geom = sim.model.geom_name2id('wall_box')
    # box_last_geom = sim.model.geom_name2id('box_part_12')
    # print(sim.model.body_name2id('wrist_3_link'))
    ctemp = np.zeros(6, dtype=np.float64)
    csum = np.zeros(6, dtype=np.float64)
    tau = np.zeros(6, dtype=np.float64)
    if sim.data.ncon > 1:
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            cond1 = contact.geom1 == gripper_geom_id
            cond2 = contact.geom2 == gripper_geom_id
            if cond1:
                # print(contact.geom2)
                if contact.geom2 == wall_geom:
                    functions.mj_contactForce(sim.model, sim.data, i, ctemp)
                    csum += ctemp
            elif cond2:
                # print(contact.geom1)
                if contact.geom1 == wall_geom:
                    functions.mj_contactForce(sim.model, sim.data, i, ctemp)
                    csum += ctemp
    gripper_orn = sim.data.get_geom_xmat(gripper_geom)
    force = np.dot(csum[0:3], gripper_orn)
    x = 0
    y = 1
    z = 2

    force = np.array([force[x], force[y], force[z]])
    torque = np.dot(-csum[3:6], gripper_orn)
    torque = np.array([torque[x], torque[y], torque[z]])
    tau = np.append(force, torque)
    # print('Tau: ', tau)
    # print('--------------------------------------------- \n')
    return tau
