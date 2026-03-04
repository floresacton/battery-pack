import control as ct
import numpy as np
import matplotlib.pyplot as plt

# LC Oscillator Analysis
# The transfer function of an LC circuit is: H(s) = 1/(LCs² + 1)
# This can be written as: H(s) = ω₀²/(s² + ω₀²)
# where ω₀ = 1/√(LC) is the resonant frequency

# Example: LC oscillator with resonant frequency at 1 rad/s
L = 1.0  # Henry
C = 1.0  # Farad
omega_0 = 1 / np.sqrt(L * C)  # Resonant frequency

print(f"LC Oscillator Parameters:")
print(f"L = {L} H")
print(f"C = {C} F")
print(f"Resonant frequency ω₀ = {omega_0:.3f} rad/s")
print(f"Resonant frequency f₀ = {omega_0 / (2 * np.pi):.3f} Hz")

# Create transfer function: H(s) = ω₀²/(s² + ω₀²)
num = [omega_0**2]
den = [1, 0, omega_0**2]
sys_lc = ct.TransferFunction(num, den)

print(f"\nTransfer Function:")
print(sys_lc)

# Create state-space representation
sys_lc_ss = ct.tf2ss(sys_lc)
print(f"\nState-Space Form:")
print(sys_lc_ss)

# Create figure with multiple subplots
fig = plt.figure(figsize=(15, 10))

# 1. Nyquist Plot
ax1 = plt.subplot(2, 2, 1)
omega = np.logspace(-2, 2, 1000)
ct.nyquist_plot(sys_lc, omega=omega, plot=True)
plt.plot(-1, 0, "rx", markersize=15, markeredgewidth=3, label="Critical Point (-1,0)")
plt.title("Nyquist Plot of LC Oscillator", fontsize=12, fontweight="bold")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.grid(True, alpha=0.3)
plt.legend()
plt.axis("equal")

# 2. Bode Magnitude Plot
ax2 = plt.subplot(2, 2, 2)
mag, phase, omega_bode = ct.bode(sys_lc, omega=omega, plot=False)
plt.semilogx(omega_bode, 20 * np.log10(mag))
plt.axvline(omega_0, color="r", linestyle="--", label=f"ω₀ = {omega_0:.2f} rad/s")
plt.title("Bode Magnitude Plot", fontsize=12, fontweight="bold")
plt.xlabel("Frequency (rad/s)")
plt.ylabel("Magnitude (dB)")
plt.grid(True, alpha=0.3)
plt.legend()

# 3. Bode Phase Plot
ax3 = plt.subplot(2, 2, 3)
plt.semilogx(omega_bode, phase * 180 / np.pi)
plt.axvline(omega_0, color="r", linestyle="--", label=f"ω₀ = {omega_0:.2f} rad/s")
plt.axhline(-90, color="g", linestyle=":", alpha=0.5, label="Phase at resonance")
plt.title("Bode Phase Plot", fontsize=12, fontweight="bold")
plt.xlabel("Frequency (rad/s)")
plt.ylabel("Phase (degrees)")
plt.grid(True, alpha=0.3)
plt.legend()

# 4. Detailed Nyquist with annotations
ax4 = plt.subplot(2, 2, 4)
# Calculate frequency response manually for annotation
s = 1j * omega
H = omega_0**2 / (s**2 + omega_0**2)
plt.plot(H.real, H.imag, "b-", linewidth=2, label="ω: 0 → +∞")
plt.plot(H.real, -H.imag, "b--", linewidth=2, label="ω: 0 → -∞")

# Mark special frequencies
freq_points = [0.1, 0.5, 1.0, 2.0, 10.0]
colors = ["red", "orange", "green", "purple", "brown"]
for f, color in zip(freq_points, colors):
    s_point = 1j * f
    H_point = omega_0**2 / (s_point**2 + omega_0**2)
    plt.plot(
        H_point.real, H_point.imag, "o", color=color, markersize=8, label=f"ω = {f:.1f}"
    )

plt.plot(-1, 0, "rx", markersize=15, markeredgewidth=3, label="Critical Point")
plt.plot(0, 0, "ko", markersize=8, label="Origin")
plt.title("Annotated Nyquist Plot", fontsize=12, fontweight="bold")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.grid(True, alpha=0.3)
plt.legend(loc="best", fontsize=8)
plt.axis("equal")
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()

# Print key observations
print("\n" + "=" * 60)
print("KEY OBSERVATIONS:")
print("=" * 60)
print(f"1. At ω = 0: H(j0) = {omega_0**2 / omega_0**2:.2f} (purely real, positive)")
print(f"2. At ω = ω₀ = {omega_0:.2f}: H(jω₀) = ∞ (pole on jω axis)")
print(f"3. At ω → ∞: H(j∞) → 0")
print(f"4. The Nyquist plot goes to infinity at the resonant frequency")
print(f"5. This creates a semicircular path that 'wraps around' infinity")
print(f"6. The system is MARGINALLY STABLE (poles on jω axis)")
print(f"7. The plot does NOT encircle (-1, 0) → stable in open loop")

# Compare with damped oscillator
print("\n" + "=" * 60)
print("COMPARISON WITH DAMPED RLC OSCILLATOR:")
print("=" * 60)
R = 0.2  # Add resistance for damping
# Transfer function: H(s) = ω₀²/(s² + 2ζω₀s + ω₀²)
zeta = R / (2 * np.sqrt(L / C))  # Damping ratio
num_rlc = [omega_0**2]
den_rlc = [1, 2 * zeta * omega_0, omega_0**2]
sys_rlc = ct.TransferFunction(num_rlc, den_rlc)

plt.figure(figsize=(10, 8))
ct.nyquist_plot(sys_lc, omega=omega, label="LC (undamped)")
ct.nyquist_plot(sys_rlc, omega=omega, label=f"RLC (ζ={zeta:.3f})")
plt.plot(-1, 0, "rx", markersize=15, markeredgewidth=3, label="Critical Point")
plt.title("Comparison: LC vs RLC Oscillator", fontsize=14, fontweight="bold")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.grid(True, alpha=0.3)
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()

print(f"\nWith damping (R = {R} Ω, ζ = {zeta:.3f}):")
print("- The Nyquist plot becomes a finite circle")
print("- No longer passes through infinity")
print("- System becomes asymptotically stable")
