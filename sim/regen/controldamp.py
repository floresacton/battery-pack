import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -------------------------
# Plant parameters
# -------------------------
Vin = 65.0
L = 4e-6
C = 100e-6
R_on = 0.5
R_off = 1e12

fsw = 25000.0
Tsw = 1.0 / fsw
t_end = 2e-3

# -------------------------
# Controller parameters
# -------------------------
I_ref = 100.0  # desired average inductor current (A)
Kp = 0.0001  # proportional gain on current error (tune)
Ki = 0.00  # integral gain (A/s -> duty accumulation) (tune if needed)
Kd_i = -0.000066  # -0.017  # virtual damping gain: duty per amp
dmin, dmax = 0.0, 0.98  # duty limits

# -------------------------
# Simulation options
# -------------------------
max_step = Tsw / 50.0  # ODE solver step limit within switch period
initial_duty = (I_ref * R_on) / (Vin + 1e-12)  # sensible feedforward initial guess
initial_state = np.array([0.0, Vin])  # [iL, vC]


# -------------------------
# ODE right-hand side
# uses a global current_duty set outside the RHS
# -------------------------
def lc_rhs_global(t_local, x_local, t0):
    iL, v = x_local
    phase = (t_local - t0) % Tsw
    R = R_on if phase < current_duty * Tsw else R_off
    diL = (Vin - v) / L
    dv = (iL - v / R) / C
    return [diL, dv]


# -------------------------
# Period-by-period simulation implementing ONLY inductor-current damping
# -------------------------
def simulate_current_damped(Kp, Ki, Kd_i, I_ref, t_end):
    t = 0.0
    x = initial_state.copy()
    duty = float(np.clip(initial_duty, dmin, dmax))
    Iint = 0.0

    tout = []
    iout = []
    vout = []
    dout = []
    Iavg_periods = []
    t_period_ends = []
    duty_history = []

    global current_duty
    current_duty = duty

    while t < t_end:
        # set the duty used during this switching period
        current_duty = duty
        t_span_end = min(t + Tsw, t_end)

        # integrate the plant over one switching period
        sol = solve_ivp(
            lambda tt, xx: lc_rhs_global(tt, xx, t),
            [t, t_span_end],
            x,
            max_step=max_step,
            rtol=1e-7,
            atol=1e-9,
        )

        # record the waveform segments
        tout.extend(sol.t.tolist())
        iout.extend(sol.y[0].tolist())
        vout.extend(sol.y[1].tolist())
        dout.extend([duty] * len(sol.t))

        # update state to the end of the period
        x = np.array([sol.y[0, -1], sol.y[1, -1]])
        t_next = sol.t[-1]

        # compute average inductor current over this period (numerical integration)
        Iavg = np.trapz(sol.y[0], sol.t) / (sol.t[-1] - sol.t[0])
        Iavg_periods.append(Iavg)
        t_period_ends.append(t_next)

        # PI controller (period-by-period)
        err = I_ref - Iavg
        Iint += err * (t_next - t)  # integrator accumulates error over the period

        duty_ff = (I_ref * R_on) / (Vin + 1e-12)  # simple feedforward

        # Inductor-current virtual damping: subtract Kd_i * Iavg
        damping_term = Kd_i * (Iavg - I_ref)  # units: duty (unitless)

        # combine control: feedforward + PI - virtual damping
        duty_cmd = duty_ff + Kp * err + Ki * Iint - damping_term

        # anti-windup & saturation
        duty_cmd_clamped = np.clip(duty_cmd, dmin, dmax)
        if duty_cmd_clamped != duty_cmd:
            # simple anti-windup: undo the integrator step if we saturated
            Iint -= err * (t_next - t)
        duty = duty_cmd_clamped

        duty_history.append(duty)
        # advance time
        t = t_next

    return (
        np.array(tout),
        np.array(iout),
        np.array(vout),
        np.array(dout),
        np.array(t_period_ends),
        np.array(Iavg_periods),
        np.array(duty_history),
    )


# -------------------------
# Run simulation with pure inductor-current damping
# -------------------------
print(f"Running sim: Kp={Kp}, Ki={Ki}, Kd_i={Kd_i} (duty per amp).")
ta, ia, va, da, tp_end, Iavgs, duties = simulate_current_damped(
    Kp, Ki, Kd_i, I_ref, t_end
)

# -------------------------
# Plot results: single figure with dual y axis (v and i), plus duty plot
# -------------------------
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(ta * 1e3, va, color="tab:red", linewidth=1.6, label="Vcap")
ax1.set_xlabel("time (ms)")
ax1.set_ylabel("Vcap (V)", color="tab:red")
ax1.tick_params(axis="y", labelcolor="tab:red")

ax2 = ax1.twinx()
ax2.plot(ta * 1e3, ia, color="tab:blue", linewidth=0.9, label="iL")
ax2.plot(tp_end * 1e3, Iavgs, "k.-", markersize=6, label="Iavg (period)")
ax2.set_ylabel("iL (A)", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")

plt.title("LC Filter — Inductor-Current Virtual Damping (pure i-feedback)")
ax1.grid(True)
fig.tight_layout()
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# duty plot
plt.figure(figsize=(10, 2.2))
plt.plot(ta * 1e3, da, linewidth=0.8)
plt.plot(
    np.linspace(0, t_end, len(duties)) * 1e3,
    duties,
    "r.-",
    markersize=4,
    label="Period duty",
)
plt.xlabel("time (ms)")
plt.ylabel("duty")
plt.title("Duty history (per-step)")
plt.grid(True)
plt.ylim(-0.05, 1.05)
plt.legend()
plt.tight_layout()

plt.show()
