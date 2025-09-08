
import numpy as np
import matplotlib.pyplot as plt
import math

# === Parameters ===
L = 330e-6      # Inductance (H)
C = 10e-6       # Output cap (F)
C_ESR = 0.01    # Cap ESR (Ω)
L_ESR = 0.3     # Inductor ESR (Ω)
VIN = 65.0      # Input voltage (V)
FSW = 200e3     # Switching frequency (Hz)

VRAMP = 1.8   # Peak to Peak ramp (V)
VREF = 1.5 # comparator refrence voltage

VOUT = 5.0 # vout

# Example loads
R_LOADS = [25.0, 40.0, 55.0]

# Frequency sweep
f_sweep = np.logspace(1, 6, 2000)   # 10 Hz to 1 MHz
w_sweep = 2 * np.pi * f_sweep
s_sweep = 1j * w_sweep

# --- Modulator gain ---
def Gmod():
    return VIN/VRAMP

# --- Plant transfer function: duty → Vout ---
def Gplant(s, r_load):
    """Duty-to-Vout including inductor ESR and cap ESR."""
    num = VIN * (1 + s * C_ESR * C)
    den = L * C * s**2 + (L_ESR * C + L / r_load) * s + 1.0
    return num / den

# --- Type III compensator ---
def Gc_type3(s, fz1, fz2, fp1, fp2, K):
    z1 = 1 + s/(2*np.pi*fz1)
    z2 = 1 + s/(2*np.pi*fz2)
    p1 = 1 + s/(2*np.pi*fp1)
    p2 = 1 + s/(2*np.pi*fp2)
    return K * (z1 * z2) / (s * p1 * p2)

# --- Resonances ---
f_lc = 1.0 / (2 * math.pi * math.sqrt(L * C))
f_rc = 1.0 / (2 * math.pi * C_ESR * C)
print(f"LC resonance f_lc = {f_lc:.1f} Hz")
print(f"Cap ESR zero f_rc = {f_rc:.1f} Hz")

# --- Compensator design points ---
# These must be equal for the part calculation below
fz1, fz2 = 2.0e3, 2.0e3     # zeros near LC resonance
fp1, fp2 = 1.0e5, 1.0e5     # poles near half fsw and a decade above crossover
K = 25    # adjust to set crossover (~15 kHz here)

# --- Properties ---
r_load = 40.0
G = Gplant(1j*2*np.pi*f_sweep, r_load) * Gc_type3(1j*2*np.pi*f_sweep, fz1, fz2, fp1, fp2, K) * Gmod()
mag = np.abs(G)
idx = np.argmin(np.abs(mag - 1))
f_crossover = f_sweep[idx]
omega_cross = 2 * np.pi * f_crossover
print(f"Plant crossover frequency ~ {f_crossover:.1f} Hz") 

comp_cs_gain = 1/(Gmod()*Gplant(1j*2*np.pi*f_crossover, r_load))
comp_cs_mag = np.abs(comp_cs_gain)
print(f"Compinsator crossover gain magnitude {comp_cs_mag:.5f}")

R_1 = 1e5

R_2 = R_1 * comp_cs_mag / np.sqrt(K)
C_1 = 1.0 / (omega_cross * R_2 * np.sqrt(K))
C_2 = np.sqrt(K) / (omega_cross * R_2)
C_3 = np.sqrt(K) / (omega_cross * R_1)
R_3 = 1.0 / (omega_cross * C_3 * np.sqrt(K))
R_4 = R_1 * VREF / (VOUT - VREF)

print(f"R_1 = {R_1:2f} Ohm")
print(f"R_2 = {R_2:2f} Ohm")
print(f"C_1 = {C_1:.3e} F")
print(f"C_2 = {C_2:3e} F")
print(f"C_3 = {C_3:3e} F")
print(f"R_3 = {R_3:2f} Ohm")
print(f"R_4 = {R_4:2f} Ohm")


# --- Graphs ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,8), sharex=True)
for R in R_LOADS:
    # Loop gain
    Gloop = Gmod() * Gplant(s_sweep, R) * Gc_type3(s_sweep, fz1, fz2, fp1, fp2, K)
    mag = 20*np.log10(np.abs(Gloop))
    phase = np.angle(Gloop, deg=True)

    # Magnitude
    ax1.semilogx(f_sweep, mag, label=f'R={R}Ω')
    ax1.axhline(0, color='k', ls=':')  # 0 dB line
    ax1.set_ylabel('Magnitude (dB)')
    ax1.grid(True, which='both', ls=':')

    # Phase
    ax2.semilogx(f_sweep, phase, label=f'R={R}Ω')
    ax2.axhline(-180, color='k', ls=':')  # -180° line
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (deg)')
    ax2.grid(True, which='both', ls=':')

# Legends
ax1.set_title('Open Loop with Type III Compensator')
ax1.legend()
ax2.legend()

plt.tight_layout()
plt.show()

# --- Plot plant only ---
# plt.figure(figsize=(9,6))
# for R in R_LOADS:
#     plt.semilogx(f_sweep, 20*np.log10(np.abs(Gplant(s_sweep, R))), label=f'Plant R={R}Ω')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude (dB)')
# plt.title('Buck Power Stage |Gvd|')
# plt.grid(True, which='both', ls=':')
# plt.legend()
# plt.tight_layout()

# --- Plot loop gain (plant * compensator) ---
# plt.figure(figsize=(9,6))
# for R in R_LOADS:
#     Gloop = Gmod() * Gplant(s_sweep, R) * Gc_type3(s_sweep, fz1, fz2, fp1, fp2, K)
#     plt.semilogx(f_sweep, 20*np.log10(np.abs(Gloop)), label=f'Loop gain R={R}Ω')
# plt.axhline(0, color='k', ls=':')  # 0 dB line
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude (dB)')
# plt.title('Open Loop with Type III Compensator')
# plt.grid(True, which='both', ls=':')
# plt.legend()
# plt.tight_layout()
# 
# plt.figure(figsize=(9,6))
# for R in R_LOADS:
#     Gloop = Gmod() * Gplant(s_sweep, R) * Gc_type3(s_sweep, fz1, fz2, fp1, fp2, K)
#     plt.semilogx(f_sweep, np.angle(Gloop, deg=True), label=f'Loop gain R={R}Ω')
# plt.axhline(-180, color='k', ls=':')  # phase margin reference
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Phase (deg)')
# plt.title('Loop Phase with Type III Compensator')
# plt.grid(True, which='both', ls=':')
# plt.legend()
# plt.tight_layout()
# 
# plt.show()
