import numpy as np
import matplotlib.pyplot as plt
import math

# === Parameters ===
L = 330e-6      # H
C = 10e-6       # F
C_ESR = 0.01 # capacitor ESR (10 mΩ)
L_ESR = 0.3       # inductor ESR (300 mΩ)
VIN = 65.0      # input voltage
FSW = 200e3     # Hz

# Example loads
R_LOADS = [25.0, 40.0, 55.0]

# Frequency sweep
f_sweep = np.logspace(1, 6, 1200)   # 10 Hz to 1 MHz
w_sweep = 2 * np.pi * f_sweep
s_sweep = 1j * w_sweep

# --- Transfer functions ---
def Gvd_with_RL(s, r_load):
    """Duty-to-Vout with inductor ESR"""
    num = VIN * (1 + s * C_ESR * C)
    den = L * C * s**2 + (L_ESR * C + L / r_load) * s + 1.0
    return num / den

# --- Resonances ---
f_lc = 1.0 / (2 * math.pi * math.sqrt(L * C))       # LC resonance
f_rc = 1.0 / (2 * math.pi * C_ESR * C)           # capacitor ESR zero

print(f"LC resonance f_lc = {f_lc:.1f} Hz")
print(f"Cap ESR zero f_rc = {f_rc:.1f} Hz")

# --- Plots ---
plt.figure(figsize=(9,6))
for R in R_LOADS:
    G_new = Gvd_with_RL(s_sweep, R)
    plt.semilogx(f_sweep, 20*np.log10(np.abs(G_new)), label=f'R_L={L_ESR}Ω, R={R}Ω')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Buck Power Stage |Gvd|')
plt.grid(True, which='both', ls=':')
plt.legend()
plt.tight_layout()

plt.figure(figsize=(9,6))
for R in R_LOADS:
    G_new = Gvd_with_RL(s_sweep, R)
    plt.semilogx(f_sweep, np.angle(G_new, deg=True), label=f'R_L={L_ESR}Ω, R={R}Ω')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (deg)')
plt.title('Buck Power Stage Phase')
plt.grid(True, which='both', ls=':')
plt.legend()
plt.tight_layout()
plt.show()
