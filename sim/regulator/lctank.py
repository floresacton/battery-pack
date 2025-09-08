import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parameters
L = 1e-3   # Inductance in H
C = 1e-6   # Capacitance in F
R = 10     # Resistance in ohms (series RLC low-pass)

# Transfer function H(s) = 1 / (LC s^2 + RC s + 1)
num = [1]                                # numerator
den = [L*C, R*C, 1]                      # denominator
system = signal.TransferFunction(num, den)

# Frequency response
w, mag, phase = signal.bode(system)

# Plot magnitude response
plt.figure()
plt.semilogx(w, mag)
plt.title("Magnitude Response of LC Circuit")
plt.xlabel("Frequency [rad/s]")
plt.ylabel("Magnitude [dB]")
plt.grid(True, which="both")

# Plot phase response
plt.figure()
plt.semilogx(w, phase)
plt.title("Phase Response of LC Circuit")
plt.xlabel("Frequency [rad/s]")
plt.ylabel("Phase [degrees]")
plt.grid(True, which="both")

plt.show()
