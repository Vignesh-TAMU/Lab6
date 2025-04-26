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
oversample_factor = 25
fs_oversampled = fs * oversample_factor  # 5 GHz
num_samples_oversampled = num_samples * oversample_factor

# Generate time vectors
t = np.arange(num_samples) / fs  # Time vector for ADC (500 MHz)
t_oversampled = np.arange(num_samples_oversampled) / fs_oversampled  # Time vector for oversampled input (5 GHz)

# Generate input signal (200 MHz tone)
Vin_oversampled = (Vfs / 2) * np.sin(2 * np.pi * f_tone * t_oversampled)  # Oversampled input for plotting
Vin = (Vfs / 2) * np.sin(2 * np.pi * f_tone * t)  # Input at 500 MHz for ADC processing

# MDAC Stage Model (Ideal)
def mdac_stage(vin, vfs):
    # Comparator levels for 2.5-bit stage (6 levels, 7 regions)
    # Input range: -Vfs/2 to Vfs/2 (-0.5V to 0.5V)
    # Levels at ±5/12, ±3/12, ±1/12 of Vfs/2
    levels = np.array([-5/12, -3/12, -1/12, 1/12, 3/12, 5/12]) * (vfs / 2)
    digital_output = np.zeros_like(vin, dtype=int)
    
    # Quantize input to 7 regions (0 to 6)
    for i in range(len(vin)):
        if vin[i] <= levels[0]:
            digital_output[i] = 0  # -3
        elif vin[i] <= levels[1]:
            digital_output[i] = 1  # -2
        elif vin[i] <= levels[2]:
            digital_output[i] = 2  # -1
        elif vin[i] <= levels[3]:
            digital_output[i] = 3  # 0
        elif vin[i] <= levels[4]:
            digital_output[i] = 4  # 1
        elif vin[i] <= levels[5]:
            digital_output[i] = 5  # 2
        else:
            digital_output[i] = 6  # 3
    
    # Map digital output to DAC levels (for residue calculation)
    # DAC levels should be centered around 0, scaled to Vfs/2
    dac_levels = np.array([-3/4, -2/4, -1/4, 0, 1/4, 2/4, 3/4]) * (vfs / 2)
    v_dac = dac_levels[digital_output]
    
    # Residue: Vres = G * (Vin - Vdac)
    residue = stage_gain * (vin - v_dac)
    # Clip residue to prevent overflow in subsequent stages
    residue = np.clip(residue, -vfs/2, vfs/2)
    
    # Map digital output to 2-bit values for digital alignment (-3 to 3)
    digital_values = np.array([-3, -2, -1, 0, 1, 2, 3])
    digital_out = digital_values[digital_output]
    
    return digital_out, residue

# Pipeline ADC Model
def pipeline_adc(vin, vfs, num_stages):
    v_stage = vin.copy()
    stage_outputs = []
    
    # Cascade stages
    for stage in range(num_stages):
        digital_out, residue = mdac_stage(v_stage, vfs)
        stage_outputs.append(digital_out)
        v_stage = residue  # Pass residue to next stage
    
    # Digital alignment and combination
    # Each stage contributes 2 bits, but we need to account for redundancy
    digital_word = np.zeros_like(vin, dtype=int)
    for i in range(num_stages):
        # Shift based on stage position (MSB to LSB)
        shift = (num_stages - 1 - i) * 2
        digital_word += (stage_outputs[i] * (2 ** shift)).astype(int)
    
    # Normalize to 13-bit range and clip to avoid overflow
    max_val = 2 ** (num_bits - 1) - 1  # e.g., 4095 for 13 bits
    min_val = -2 ** (num_bits - 1)     # e.g., -4096 for 13 bits
    digital_word = np.clip(digital_word, min_val, max_val)
    
    return digital_word

# Run the ADC (using the 500 MHz sampled input)
digital_output = pipeline_adc(Vin, Vfs, num_stages)

# Convert digital output back to analog for plotting
lsb = Vfs / (2 ** num_bits)
analog_output = digital_output * lsb

# Calculate SNR (ensure proper RMS calculation for sine wave)
signal_power = np.mean(Vin ** 2)  # RMS power of input sine wave
noise = analog_output - Vin
noise_power = np.mean(noise ** 2)  # RMS power of quantization noise
snr = 10 * np.log10(signal_power / noise_power)

print(f"SNR: {snr:.2f} dB")

# Plot waveforms (styled to match the provided plot, with oversampled input)
# Use only the first 1000 samples for better visibility at 500 MHz
num_display_samples = 20
# Corresponding number of oversampled points
num_display_samples_oversampled = num_display_samples * oversample_factor

t_display = t[:num_display_samples] * 1e6  # Convert to microseconds (500 MHz)
t_display_oversampled = t_oversampled[:num_display_samples_oversampled] * 1e6  # Oversampled time
vin_display_oversampled = Vin_oversampled[:num_display_samples_oversampled]  # Oversampled input
analog_out_display = analog_output[:num_display_samples]  # Quantized output at 500 MHz
# Interpolate the quantized output to the oversampled time grid for error calculation
analog_out_oversampled = np.repeat(analog_out_display, oversample_factor)  # Repeat each sample 10 times
error_display = analog_out_oversampled - vin_display_oversampled  # Quantization error

plt.figure(figsize=(10, 6))
# Plot oversampled input signal (smooth, blue line)
plt.plot(t_display_oversampled, vin_display_oversampled, color='blue', linewidth=0.5, label='Input Signal (Oversampled)')
# Plot quantized output as orange dashed line (at 500 MHz)
plt.plot(t_display, analog_out_display, color='orange', linestyle='--', label='Quantized Output')
# Plot quantization error as green line
#plt.plot(t_display_oversampled, error_display, color='green', linestyle='-', alpha=0.5, label='Quantization Error')
plt.xlabel('Time (s) 1e-6')
plt.ylabel('Amplitude (V)')
plt.title('Input Signal (Oversampled) vs. Quantized Output with Error')
plt.legend()
plt.grid(True)
plt.savefig('waveforms_corrected_snr.png')
