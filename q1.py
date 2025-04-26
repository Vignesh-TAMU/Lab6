import numpy as np
import matplotlib.pyplot as plt

# ADC Parameters
fs = 500e6  # Sampling frequency: 500 MHz
f_tone = 200e6  # Input tone frequency: 200 MHz
Vfs = 1.0  # Full-scale voltage: 1V
num_bits = 13  # Total ADC resolution: 13 bits
num_stages = 6  # Number of stages
bits_per_stage = 2.5  # 2.5 bits per stage (2 bits + redundancy)
stage_gain = 4  # Gain of each stage (2^2 for 2 bits)
num_samples = 8192  # Number of samples for simulation

# Oversampling for input signal visualization (10x the ADC sampling rate)
oversample_factor = 20
fs_oversampled = fs * oversample_factor  # 5 GHz
num_samples_oversampled = num_samples * oversample_factor

# Generate input signal (200 MHz tone)
t = np.arange(num_samples) / fs
t_oversampled = np.arange(num_samples_oversampled) / fs_oversampled  # Time vector for oversampled input (5 GHz)

Vin_oversampled = (Vfs / 2) * np.sin(2 * np.pi * f_tone * t_oversampled)  # Oversampled input for plotting
Vin = (Vfs / 2) * np.sin(2 * np.pi * f_tone * t)  # Input signal: Vfs/2 * sin(2πft)

# MDAC Stage Model (Ideal)
def mdac_stage(vin, vfs, bits, gain):
    # Comparator levels for 2.5-bit stage (6 levels)
    levels = np.array([-5/8, -3/8, -1/8, 1/8, 3/8, 5/8]) * vfs
    digital_output = np.zeros_like(vin, dtype=int)
    
    # Quantize input to 6 levels (2.5 bits)
    for i in range(len(vin)):
        if vin[i] <= levels[0]:
            digital_output[i] = 0
        elif vin[i] <= levels[1]:
            digital_output[i] = 1
        elif vin[i] <= levels[2]:
            digital_output[i] = 2
        elif vin[i] <= levels[3]:
            digital_output[i] = 3
        elif vin[i] <= levels[4]:
            digital_output[i] = 4
        elif vin[i] <= levels[5]:
            digital_output[i] = 5
        else:
            digital_output[i] = 6
    
    # Convert digital output back to analog for residue
    dac_levels = np.array([-6/8, -4/8, -2/8, 0, 2/8, 4/8, 6/8]) * vfs
    v_dac = dac_levels[digital_output]
    
    # Residue: Vres = G * (Vin - Vdac)
    residue = gain * (vin - v_dac)
    
    return digital_output, residue

# Pipeline ADC Model
def pipeline_adc(vin, vfs, num_stages, bits_per_stage, stage_gain):
    v_stage = vin.copy()
    stage_outputs = []
    
    # Cascade stages
    for stage in range(num_stages):
        digital_out, residue = mdac_stage(v_stage, vfs, bits_per_stage, stage_gain)
        stage_outputs.append(digital_out)
        v_stage = residue  # Pass residue to next stage
    
    # Digital alignment and combination (corrected)
    digital_word = np.zeros_like(vin, dtype=int)
    for i in range(num_stages):
        # Each stage contributes 2 bits (since 2.5-bit stage resolves 2 bits after redundancy)
        # Shift based on stage position: stage 0 contributes to MSB, stage 5 to LSB
        shift = (num_stages - 1 - i) * 2
        # Adjust digital output to account for the 2.5-bit encoding (0 to 6) to proper 2-bit values (0 to 3)
        adjusted_digital = stage_outputs[i] - 3  # Center around 0 (e.g., 0 to 6 -> -3 to 3)
        digital_word += (adjusted_digital * (2 ** shift)).astype(int)
    
    # Normalize to 13-bit range and clip to avoid overflow
    max_val = 2 ** (num_bits - 1) - 1  # e.g., 4095 for 13 bits
    min_val = -2 ** (num_bits - 1)     # e.g., -4096 for 13 bits
    digital_word = np.clip(digital_word, min_val, max_val)
    
    return digital_word

# Run the ADC
digital_output = pipeline_adc(Vin, Vfs, num_stages, bits_per_stage, stage_gain)

# Convert digital output back to analog for plotting
lsb = Vfs / (2 ** num_bits)
analog_output = digital_output * lsb

# Calculate SNR
signal_power = np.mean(Vin ** 2)
noise = analog_output - Vin
noise_power = np.mean(noise ** 2)
snr = 10 * np.log10(signal_power / noise_power)

print(f"SNR: {snr:.2f} dB")

# Plot waveforms (styled to match the provided plot)
# Use only the first 1000 samples for better visibility
num_display_samples = 10
# Corresponding number of oversampled points
num_display_samples_oversampled = num_display_samples * oversample_factor

t_display = t[:num_display_samples] * 1e6  # Convert to microseconds
t_display_oversampled = t_oversampled[:num_display_samples_oversampled] * 1e6  # Oversampled time
vin_display_oversampled = Vin_oversampled[:num_display_samples_oversampled]  # Oversampled input
vin_display = Vin[:num_display_samples]
analog_out_display = analog_output[:num_display_samples]

plt.figure(figsize=(10, 6))
# Plot input signal as dense bars (simulating the blue "filled" look)
plt.plot(t_display, vin_display, color='blue', linewidth=0.5, label='Input Signal')
# Plot oversampled input signal (smooth, blue line)
plt.plot(t_display_oversampled, vin_display_oversampled, color='red', linewidth=0.5, label='Input Signal (Oversampled)')
# Plot quantized output as orange dashed line
plt.plot(t_display, analog_out_display, color='orange', linestyle='--', label='Quantized Output')
plt.xlabel('Time (s) 1e-6')
plt.ylabel('Amplitude (V)')
plt.title('Input Signal vs. Quantized Output')
plt.legend()
plt.grid(True)
plt.savefig('waveforms_corrected.png')
