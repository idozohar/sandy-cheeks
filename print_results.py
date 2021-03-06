import UR5_kinematics as UR_kin
import numpy as np
from matplotlib import pyplot as plt
import filters
import mujoco_py
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


def print_q(q1, q2, dt, headline):
    q = np.concatenate((q1, q2), axis=1)
    t_len = len(q[0, :]) * dt
    t = np.linspace(0, t_len, len(q[0, :]))

    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex='col')
    if headline == 'angle':
        ax1.set_title('desired joints angle[rad]')
    elif headline == 'speed':
        ax1.set_title('desired joints speed[rad/sec]')

    ax1.plot(t, q[0, :])
    ax1.set_ylabel('q1')
    ax2.plot(t, q[1, :])
    ax2.set_ylabel('q2')
    ax3.plot(t, q[2, :])
    ax3.set_ylabel('q3')
    ax4.plot(t, q[3, :])
    ax4.set_ylabel('q4')
    ax5.plot(t, q[4, :])
    ax5.set_ylabel('q5')
    ax6.plot(t, q[5, :])
    ax6.set_ylabel('q6')
    ax6.set_xlabel('Time [sec]')
    plt.show()


def print_q_actual(q1, q2, q_actual, dt, headline):
    q = np.concatenate((q1, q2), axis=1)
    t_len = len(q_actual[0, :]) * dt
    t = np.linspace(0, t_len, len(q_actual[0, :]))

    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex='col')
    if headline == 'angle':
        ax1.set_title('desired joints angle[rad]')
    elif headline == 'speed':
        ax1.set_title('desired joints speed[rad/sec]')
    print(q_actual[0, :])
    ax1.plot(t, q[0, 0:len(q_actual[0, :])], t, q_actual[0, :])
    ax1.set_ylabel('q1')
    ax1.grid(axis='both')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax2.plot(t, q[1, 0:len(q_actual[0, :])], t, q_actual[1, :])
    ax2.set_ylabel('q2')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax2.grid(axis='both')
    ax3.plot(t, q[2, 0:len(q_actual[0, :])], t, q_actual[2, :])
    ax3.set_ylabel('q3')
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax3.grid(axis='both')
    ax4.plot(t, q[3, 0:len(q_actual[0, :])], t, q_actual[3, :])
    ax4.set_ylabel('q4')
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax4.grid(axis='both')
    ax5.plot(t, q[4, 0:len(q_actual[0, :])], t, q_actual[4, :])
    ax5.set_ylabel('q5')
    ax5.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax5.grid(axis='both')
    ax6.plot(t, q[5, 0:len(q_actual[0, :])], t, q_actual[5, :])
    ax6.set_ylabel('q6')
    ax6.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax6.grid(axis='both')
    ax6.set_xlabel('Time [sec]')
    plt.show()


def print_pos_error(q, dt):
    t_len = len(q[0, :]) * dt
    t = np.linspace(0, t_len, len(q[0, :]))
    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex='col')
    ax1.set_title('Joints Position Error[rad]')
    ax1.plot(t, q[0, :])
    ax1.set_ylabel('q1')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax1.grid(axis='both')
    ax2.plot(t, q[1, :])
    ax2.set_ylabel('q2')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax2.grid(axis='both')
    ax3.plot(t, q[2, :])
    ax3.set_ylabel('q3')
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax3.grid(axis='both')
    ax4.plot(t, q[3, :])
    ax4.set_ylabel('q4')
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax4.grid(axis='both')
    ax5.plot(t, q[4, :])
    ax5.set_ylabel('q5')
    ax5.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax5.grid(axis='both')
    ax6.plot(t, q[5, :])
    ax6.set_ylabel('q6')
    ax6.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax6.grid(axis='both')
    ax6.set_xlabel('Time [sec]')
    plt.show()


def print_force(f_in_log, dt):
    t_len = len(f_in_log[:]) * dt
    t = np.linspace(0, t_len, len(f_in_log[:]))
    plt.figure()
    plt.plot(t, f_in_log[:])
    plt.title('Force [N?]')
    plt.grid(axis='both')
    plt.ylabel('f_in')
    plt.xlabel('Time [sec]')
    plt.show()


def print_torque(u_log, dt):
    t_len = len(u_log[0, :]) * dt
    t = np.linspace(0, t_len, len(u_log[0, :]))
    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex='col')
    ax1.set_title('Torque [Nm]')
    ax1.plot(t, u_log[0, :])
    ax1.grid(axis='both')
    ax1.set_ylabel('q1')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax2.plot(t, u_log[1, :])
    ax2.set_ylabel('q2')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax2.grid(axis='both')
    ax3.plot(t, u_log[2, :])
    ax3.set_ylabel('q3')
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax3.grid(axis='both')
    ax4.plot(t, u_log[3, :])
    ax4.set_ylabel('q4')
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax4.grid(axis='both')
    ax5.plot(t, u_log[4, :])
    ax5.set_ylabel('q5')
    ax5.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax5.grid(axis='both')
    ax6.plot(t, u_log[5, :])
    ax6.set_ylabel('q6')
    ax6.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax6.grid(axis='both')
    ax6.set_xlabel('Time [sec]')
    plt.show()


def clamp(num, min_value, max_value):
    return max(min(num, max_value), min_value)


def force_scope(sim, force_scope_log):
    forces = sim.data.cfrc_ext[sim.model.body_name2id('OUR_TABLE')]
    forces = np.reshape(forces, (6, 1))
    force_scope_log = np.append(force_scope_log, forces, axis=1)
    # filtered_force_log = my_filter.butter_lowpass_filter(force_scope_log, 5, 15, 2)  # 8, 20, 2
    return force_scope_log


def print_force_scope(f_log, dt):
    t_len = len(f_log[0, :]) * dt
    t = np.linspace(0, t_len, len(f_log[0, :]))
    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex='col')
    ax1.set_title('Force and Torque')
    ax1.plot(t, f_log[0, :])
    plt.grid(axis='both')
    ax1.set_ylabel('Torque x')
    ax2.plot(t, f_log[1, :])
    ax2.set_ylabel('Torque y')
    ax3.plot(t, f_log[2, :])
    ax3.set_ylabel('Torque z')
    ax4.plot(t, f_log[3, :])
    ax4.set_ylabel('Force x')
    ax5.plot(t, f_log[4, :])
    ax5.set_ylabel('Force y')
    ax6.plot(t, f_log[5, :])
    ax6.set_ylabel('Force z')
    ax6.set_xlabel('Time [sec]')
    plt.show()


def print_force_scope2(force_log, dt):
    t_len = len(force_log[0, :]) * dt
    force_log_filt = filters.butter_lowpass_filter(force_log, 3.6, 30, 6)
    t = np.linspace(0, t_len, len(force_log[0, :]))
    f_log_mean = np.zeros((6, 1))
    step = 100
    for i in range(len(force_log_filt[0])-1):
        if i >= len(force_log_filt[0]) - step:
            f_mean = np.mean(force_log_filt[:, (i - step):i], axis=1)
        else:
            f_mean = np.mean(force_log_filt[:, i:(i + step)], axis=1)
        f_mean = np.reshape(f_mean, (6, 1))
        f_log_mean = np.append(f_log_mean, f_mean, axis=1)

    # print("x_min:", np.min(f_log_mean[0]), "x_max:", np.max(f_log_mean[0]))
    # print("y_min:", np.min(f_log_mean[1]), "y_max:", np.max(f_log_mean[1]))
    # print("z_min:", np.min(f_log_mean[2]), "z_max:", np.max(f_log_mean[2]))
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col')
    ax1.set_title('Force [N]')
    ax1.plot(t, force_log_filt[0, :], t, f_log_mean[0, :])
    ax1.grid(axis='both')
    ax1.set_ylabel('Force x')
    ax2.plot(t, force_log_filt[1, :], t, f_log_mean[1, :])
    ax2.set_ylabel('Force y')
    ax2.grid(axis='both')
    ax3.plot(t, force_log_filt[2, :], t, f_log_mean[2, :])
    ax3.set_ylabel('Force z')
    ax3.grid(axis='both')
    ax3.set_xlabel('Time [sec]')
    plt.show()


def contacts(sim):
    print('number of contacts', sim.data.ncon)
    for i in range(sim.data.ncon):
        # Note that the contact array has more than `ncon` entries,
        # so be careful to only read the valid entries.
        contact = sim.data.contact[i]
        print('contact', i)
        print('dist', contact.dist)
        print('geom1', contact.geom1, sim.model.geom_id2name(contact.geom1))
        print('geom2', contact.geom2, sim.model.geom_id2name(contact.geom2))
        # There's more stuff in the data structure
        # See the mujoco documentation for more info!
        geom2_body = sim.model.geom_bodyid[sim.data.contact[i].geom2]
        print('body: ', geom2_body)
        print(' Contact force on geom2 body', sim.data.cfrc_ext[geom2_body])
        print('norm', np.sqrt(np.sum(np.square(sim.data.cfrc_ext[geom2_body]))))
        # Use internal functions to read out mj_contactForce
        c_array = np.zeros(6, dtype=np.float64)
        print('c_array', c_array)
        mujoco_py.functions.mj_contactForce(sim.model, sim.data, i, c_array)
        print('c_array', c_array)
        print(' ')
    print('------------------------------------------------------- \n \n')

# ------------------------------------------------------------------------#
def print_path_xyz(q_path):
    print_len = len(q_path[1, :])
    full_path = np.zeros((print_len, 4, 4))
    xyz_path = full_path[0, 0:3, -1]
    xyz_path = np.reshape(xyz_path, (1, 3))
    for k in range(1, print_len):
        full_path = UR_kin.forward(q_path[:, k])
        full_path = np.reshape(full_path[0:3, 3], (1, 3))
        xyz_path = np.concatenate((xyz_path, full_path), axis=0)

    # print_xyz(xyz_path)


def print_xyz_old(path):

    plt.subplot(231)
    plt.plot(path[0, 0], path[0, 1], 'xb')
    plt.plot(path[149, 0], path[149, 1], 'xg')
    plt.plot(path[299, 0], path[299, 1], 'xm')
    plt.plot(path[:, 0], path[:, 1], 'r')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.subplot(232)
    plt.plot(path[0, 0], path[0, 2], 'xb')
    plt.plot(path[149, 0], path[149, 2], 'xg')
    plt.plot(path[299, 0], path[299, 2], 'xm')
    plt.plot(path[:, 0], path[:, 2], 'r')
    plt.xlabel('X')
    plt.ylabel('Z')

    plt.subplot(233)
    plt.plot(path[0, 1], path[0, 2], 'xb')
    plt.plot(path[149, 1], path[149, 2], 'xg')
    plt.plot(path[299, 1], path[299, 2], 'xm')
    plt.plot(path[:, 1], path[:, 2], 'r')
    plt.xlabel('Y')
    plt.ylabel('Z')

    plt.suptitle('XYZ')
    plt.subplots_adjust(top=0.92, bottom=0, left=0.20, right=0.95, hspace=0.25, wspace=0.35)
    plt.show()

    return 1

def print_xyz(x_r, x_im, x_0, dt):

    t_len = len(x_r[0, :]) * dt
    im_len = len(x_im[0, :])
    t = np.linspace(0, t_len, len(x_r[0, :]))
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex='col')

    ax1.set_title('X')
    ax1.plot(t, x_r[0, :],'r', t[-im_len-1:-1], x_im[0, :],'--b', t, x_0[0, :],'-.g')
    ax1.grid(axis='both')
    ax1.set_ylabel('x')
    ax1.set_xlabel('Time [sec]')

    ax2.set_title('Y')
    ax2.plot(t, x_r[1, :],'r', t[-im_len-1:-1], x_im[1, :],'--b', t, x_0[1, :],'-.g')
    ax2.set_ylabel('y')
    ax2.set_xlabel('Time [sec]')
    ax2.grid(axis='both')

    ax3.set_title('Z')
    ax3.plot(t, x_r[2, :],'r', t[-im_len-1:-1], x_im[2, :],'--b', t, x_0[2, :],'-.g')
    ax3.set_ylabel('z')
    ax3.set_xlabel('Time [sec]')
    ax3.grid(axis='both')

    plt.legend(['X_r', 'X_im', 'X_0'])
    plt.show()

def print_difference(x_r, x_im, x_0, dt):

    t_len = len(x_r[0, :]) * dt
    im_len = len(x_im[0, :])
    t = np.linspace(0, t_len, len(x_r[0, :]))
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex='col')

    ax1.set_title('X')
    ax1.plot(t[-im_len-1:-1], x_r[0,-im_len-1:-1] - x_im[0, :],'r',t[-im_len-1:-1], x_im[0,:] - x_0[0,-im_len-1:-1])
    ax1.grid(axis='both')
    ax1.set_xlabel('Time [sec]')

    ax2.set_title('Y')
    ax2.plot(t[-im_len-1:-1], x_r[1,-im_len-1:-1] - x_im[1, :],'r',t[-im_len-1:-1], x_im[1,:] - x_0[1,-im_len-1:-1])
    ax2.set_xlabel('Time [sec]')
    ax2.grid(axis='both')

    ax3.set_title('Z')
    ax3.plot(t[-im_len-1:-1], x_r[2,-im_len-1:-1] - x_im[2, :],'r',t[-im_len-1:-1], x_im[2,:] - x_0[2,-im_len-1:-1])
    ax3.set_xlabel('Time [sec]')
    ax3.grid(axis='both')

    plt.legend(['X_r - X_im', 'X_im - X_0'])
    plt.show()

def print_xyz_3D(x_r):

    z0 = 0 #0.95863
    x = np.arange(-0.5, 2, 0.1)
    y = np.arange(-0.5, 2, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = 0*X
    ax1 = plt.axes(projection='3d')
    ax1.plot3D(x_r[0, :], x_r[1, :], x_r[2, :]-z0)
    ax1.scatter(x_r[0, 0], x_r[1, 0], x_r[2, 0] - z0, color='g')
    ax1.scatter(x_r[0, -1], x_r[1, -1], x_r[2, -1] - z0, color='r')
    ax1.plot_surface(X, Y, Z)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    # plt.show()

    f, (ax2, ax3, ax4) = plt.subplots(1, 3, sharex='col')
    ax2.plot(x_r[0, :], x_r[1, :])
    ax2.set_title('X-Y')
    ax2.set_ylabel('Y')
    ax2.set_xlabel('X')
    ax2.grid(axis='both')

    ax3.plot(x_r[0, :], x_r[2, :]- z0)
    ax3.set_title('X-Z')
    ax3.set_ylabel('Z')
    ax3.set_xlabel('X')
    ax3.grid(axis='both')

    ax4.plot(x_r[1, :], x_r[2, :]- z0)
    ax4.set_title('Y-Z')
    ax4.set_ylabel('Z')
    ax4.set_xlabel('Y')
    ax4.grid(axis='both')

    plt.show()

def print_xyz_time(t, x_0, x_m, x_r):

    plt.figure()
    plt.plot(t, x_0[0, :], t, x_m[0, :], t, x_r[0, :])
    plt.title('X location')
    plt.ylabel('X [m]')
    plt.xlabel('t [sec]')
    plt.grid(axis='both')
    plt.legend(['x0', 'xm', 'xr'])

    plt.figure()
    plt.plot(t, x_0[1, :], t, x_m[1, :], t, x_r[1, :])
    plt.title('Y location')
    plt.ylabel('Y [m]')
    plt.xlabel('t [sec]')
    plt.grid(axis='both')
    plt.legend(['x0', 'xm', 'xr'])

    plt.figure()
    plt.plot(t, x_0[2, :], t, x_m[2, :], t, x_r[2,:])
    plt.title('Z location')
    plt.ylabel('Z [m]')
    plt.xlabel('t [sec]')
    plt.grid(axis='both')
    plt.legend(['x0', 'xm', 'xr'])

    plt.show()