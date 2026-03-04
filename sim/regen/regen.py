# - L - C - R
#       |   S
#      GND GND

# averaged model R = 0.5/D

# state model
# L * di/dt = vin - v
# ON:  C * dv/dt = i - v/R
# OFF: C * dv/dt = i
# avg: C * dv/dt = i - gavg * v

# gavg = D/R
# D = u (input)

# states are i and v, input u (D)
# di/dt = (vin - v) / L
# dv/dt = (i - v*u/R) / C

# plan:
# gain schedule based on pack voltage
# assume that for v*u/R, v is steady enough <10%

# linearized model:
# di/dt = -v/L + 0*i + vin/L + 0*u
# dv/dt = 0*v + i/C + 0 + vin*u/R/C

import numpy as np
from scipy.signal import place_poles, StateSpace, bode
import math
import matplotlib.pyplot as plt
from scipy import signal
import control as ct


VIN_MIN = 40.0
VIN_MAX = 68.0

R = 0.5
L = 20e-6
C = 200e-6

VIN = 65.0


def system(l, c, r, vin):
    A = np.array([[0, -1 / l], [1 / c, 0]])
    B = np.array([[0], [-vin / (r * c)]])
    Cmeas = np.array([[0.0, 1.0]])  # used by observer
    Cfbk = np.array([[1.0, 0.0]])  # used by feedback

    # d does not effect poles because steady state offset
    d = np.array([[vin / l], [0.0]])
    D = np.array([[0.0]])

    sys_open = ct.ss(A, B, Cfbk, D)
    w = np.logspace(0, 6, 1000)  # 10^2 to 10^6 1000 steps

    T = 0.5
    alpha = 0.1  # α > 1 gives LAG
    num = [T, 1.0]
    den = [alpha * T, 1.0]
    C_boost_tf = ct.tf(num, den)

    T = 0.005
    alpha = 10.0  # α > 1 gives LAG
    num = [T, 1.0]
    den = [alpha * T, 1.0]
    C_lag_tf = ct.tf(num, den)

    num = [0.0, 4.0]
    den = [1.0, 0.0]
    C_integ_tf = ct.tf(num, den)
    C_integ_ss = ct.ss(C_integ_tf)

    tau = 0.0001
    num = [1.0]
    den = [tau, 1.0]
    C_LP_tf = ct.tf(num, den)
    C_LP_ss = ct.ss(C_LP_tf)

    omega0 = 15000  # notch at 20 kHz
    Q = 0.01  # quality factor
    num = [tau, 1.0]  # s^2 + ω0^2
    den = [1.0]  # s^2 + (ω0/Q)s + ω0^2
    C_notch_tf = ct.tf(num, den)

    C_total_tf = C_lag_tf * C_integ_tf * C_boost_tf
    sys_loop = C_total_tf * sys_open
    sys_closed = ct.feedback(sys_loop, 1)

    if True:
        ct.bode_plot(sys_loop, w)
        plt.suptitle("Bode Plot")
        plt.tight_layout()
        plt.show()

    if True:
        ct.nyquist_plot(sys_loop)
        plt.suptitle("Nyquist Plot")
        plt.tight_layout()
        plt.show()

    # disk margins
    DM, DGM, DPM = ct.disk_margins(sys_loop, w)

    # Print key results
    print("SISO Disk Margins:")
    print(f"  Min Gain Margin: {np.min(DGM):.3f} dB")
    print(f"  Min Phase Margin: {np.min(DPM):.2f} deg")
    print(f"  Min Disk Margin (radius): {np.min(DM):.3f}")

    # Luenberger observer
    f_des1 = 4500.0
    f_des2 = 5000.0
    s_des1 = -2 * math.pi * f_des1
    s_des2 = -2 * math.pi * f_des2
    desired_poles = [s_des1, s_des2]

    pp = place_poles(A.T, Cmeas.T, desired_poles)
    L_gain = pp.gain_matrix.T

    # simulation
    fsw = 25000.0
    Tsw = 1.0 / fsw
    dt = 100e-9
    Tsim = 0.01

    N = int(Tsim / dt)
    N_obsv = int(Tsw / dt)

    x = np.array([[0.0], [65.0]])
    x_hat = np.array([[0.0], [65.0]])
    u = 0.0

    # logs
    ts = []
    i_log = []
    v_log = []
    i_hat_log = []
    v_hat_log = []

    C_total_ss = ct.ss(C_total_tf)

    Ac = C_total_ss.A
    Bc = C_total_ss.B
    Cc = C_total_ss.C
    Dc = C_total_ss.D

    x_c = np.zeros((Ac.shape[0], 1))

    dt_ctrl = dt * N_obsv

    yadd = 0.0
    ysamples = 0.0
    for k in range(N - 1):
        pwm_on = (k * dt % Tsw) < u * Tsw

        i = x[0, 0]
        v = x[1, 0]

        di = (vin - v) / l
        if pwm_on:
            dv = (i - v / r) / c
        else:
            dv = (i) / c

        x[0, 0] = x[0, 0] + di * dt
        x[1, 0] = x[1, 0] + dv * dt

        # real system
        # x_dot = A @ x + B * u + d

        # x = x + x_dot * dt

        # measured output (real v)
        yadd += x[1, 0]
        ysamples += 1.0

        # observer
        if k % N_obsv == 0:
            cur_set = 50.0
            # feedforward steady state
            # u_steady = cur_set * r / vin

            # error = x - np.array([[cur_set], [vin]])
            # u_cont = u_steady  # - K @ error
            # u = np.clip(u_cont, 0, 1)
            # u = duty(k * dt)V

            y = [yadd / ysamples]
            yadd = 0.0
            ysamples = 0.0
            y_hat = Cmeas @ x_hat
            x_hat_dot = A @ x_hat + B * u + d + L_gain @ (y - y_hat)
            x_hat = x_hat + x_hat_dot * dt * N_obsv

            error = np.array([[cur_set - x_hat[0, 0]]])

            # --- controller state derivative ---
            x_c_dot = Ac @ x_c + Bc @ error

            # --- integrate controller state (Euler) ---
            x_c = x_c + x_c_dot * dt_ctrl

            # --- controller output ---
            u_cont = float((Cc @ x_c + Dc @ error)[0, 0])

            # --- saturate for PWM duty ---
            u = np.clip(u_cont, 0.0, 1.0)

        # log
        ts.append(k * dt)
        i_log.append(x[0, 0])
        v_log.append(x[1, 0])
        i_hat_log.append(x_hat[0, 0])
        v_hat_log.append(x_hat[1, 0])

    # --- Plot results ---
    plt.figure()
    plt.plot(ts, i_log, label="i true")
    plt.plot(ts, i_hat_log, "--", label="i_hat")
    plt.legend()
    plt.xlabel("time (s)")
    plt.ylabel("current (A)")

    plt.figure()
    plt.plot(ts, v_log, label="v true")
    plt.plot(ts, v_hat_log, "--", label="v_hat")
    plt.legend()
    plt.xlabel("time (s)")
    plt.ylabel("voltage (V)")

    plt.show()


system(L, C, R, 65.0)
