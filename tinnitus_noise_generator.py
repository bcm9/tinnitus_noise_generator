import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import sounddevice as sd

########################################################################################################
# Generate white noise
########################################################################################################
def generate_white_noise(duration_sec, sample_rate=44100):
    """
    Generate a white noise signal.
    
    Parameters:
    duration_sec (float): Duration of the noise in seconds
    sample_rate (int): Sampling rate (e.g., 44100 samples per second)
    
    Returns:
    np.array: The generated white noise signal
    """
    # Generate white noise: random values between -1 and 1
    noise = np.random.uniform(-1, 1, int(duration_sec * sample_rate))
    return noise

########################################################################################################
# Generate notched noise (white noise with a frequency notch)
########################################################################################################
def generate_notched_noise(frequency_notch, bandwidth, duration_sec, sample_rate=44100):
    """
    Generate a notched noise signal (white noise with a specific frequency range removed).
    
    Parameters:
    frequency_notch (float): The centre of the frequency to notch out (in Hz)
    bandwidth (float): The bandwidth of the notch (in Hz)
    duration_sec (float): Duration of the noise in seconds
    sample_rate (int): Sampling rate
    
    Returns:
    np.array: The generated notched noise signal
    """
    # Generate white noise
    noise = generate_white_noise(duration_sec, sample_rate)
    
    # Apply a notch filter to the white noise
    # Fourier transform to the frequency domain
    freqs = np.fft.fftfreq(len(noise), 1/sample_rate)
    noise_fft = np.fft.fft(noise)
    
    # Create a notch filter in the frequency domain
    notch_filter = np.ones_like(noise_fft)
    notch_filter[(freqs > (frequency_notch - bandwidth/2)) & (freqs < (frequency_notch + bandwidth/2))] = 0
    
    # Apply the filter by multiplying in the frequency domain
    notched_noise_fft = noise_fft * notch_filter
    
    # Inverse Fourier transform to get the filtered signal back in the time domain
    notched_noise = np.fft.ifft(notched_noise_fft)
    
    # Return the real part (ignore small imaginary components due to numerical error)
    return np.real(notched_noise)

from scipy.signal import butter, lfilter

########################################################################################################
# Generate band-pass noise (noise band) with given bandwidth and centre frequency
########################################################################################################
def generate_bandpass_noise(centre_freq, bandwidth, duration_sec, sample_rate=44100):
    """
    Generate a band-pass noise signal (noise within a specific frequency band).
    
    Parameters:
    centre_freq (float): Centre frequency of the noise band (in Hz)
    bandwidth (float): The bandwidth of the noise (in Hz)
    duration_sec (float): Duration of the noise in seconds
    sample_rate (int): Sampling rate (e.g., 44100 samples per second)
    
    Returns:
    np.array: The generated band-pass noise signal
    """
    # Generate white noise first
    white_noise = generate_white_noise(duration_sec, sample_rate)
    
    # Define the low and high cut-off frequencies for the band-pass filter
    lowcut = max(0, centre_freq - bandwidth / 2)
    highcut = min(sample_rate / 2, centre_freq + bandwidth / 2)
    
    # Design a Butterworth band-pass filter
    nyquist = 0.5 * sample_rate  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(N=4, Wn=[low, high], btype='band')
    
    # Apply the filter to the white noise
    bandpass_noise = lfilter(b, a, white_noise)
    
    return bandpass_noise

########################################################################################################
# Play audio signal directly in Python
########################################################################################################
def play_sound(signal, sample_rate=44100):
    """
    Play the given audio signal using sounddevice.
    
    Parameters:
    signal (np.array): The audio signal to play
    sample_rate (int): The sample rate of the audio (default is 44100 Hz)
    """
    # Ensure that the signal is in a floating-point format (if not already)
    if signal.dtype != np.float32:
        signal = signal.astype(np.float32)
    
    # Play the signal
    sd.play(signal, samplerate=sample_rate)
    
    # Wait until the sound finishes playing
    sd.wait()

########################################################################################################
# Play signal
########################################################################################################
# Simulation parameters
sample_rate = 44100  # 44.1 kHz sampling rate
duration_sec = 1  # seconds of noise
frequency_notch = 2000  # Tinnitus frequency in Hz
notch_bandwidth = 500  # Bandwidth of the notch in Hz

# Generate white noise and notched noise
white_noise = generate_white_noise(duration_sec, sample_rate)
notched_noise = generate_notched_noise(frequency_notch, notch_bandwidth, duration_sec, sample_rate)

# Generate band-pass noise
centre_freq = 2000  # Centre frequency of the noise band (in Hz)
bandwidth = 500  # Bandwidth of the noise band (in Hz)
bandpass_noise = generate_bandpass_noise(centre_freq, bandwidth, duration_sec)

# Assuming you already generated the `bandpass_noise` signal:
# play_sound(bandpass_noise)

# Save generated sounds as WAV files
folder = 'C:/Users/bc22/OneDrive/Documents/code/tinnitus_sound_generator/'
# write(folder+'white_noise.wav', sample_rate, white_noise.astype(np.float32))
# write('notched_noise.wav', sample_rate, notched_noise.astype(np.float32))

########################################################################################################
# Plot signals
########################################################################################################
plt.rcParams['font.family'] = 'Calibri'
# Plot the noises
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
time = np.linspace(0, duration_sec, int(sample_rate * duration_sec))

# Plot white noise
axs[0].plot(time, white_noise, color='b')
axs[0].set_title('White Noise', fontsize=18, fontweight='bold')
axs[0].set_xlabel(' ', fontsize=18, fontweight='bold')
axs[0].set_ylabel('Amplitude', fontsize=18, fontweight='bold')
axs[0].grid(True, linestyle='--', alpha=0.3)

# Plot notched noise
axs[1].plot(time, notched_noise, color='g')
axs[1].set_title('Notched Noise', fontsize=18, fontweight='bold')
axs[1].set_xlabel('Time (s)', fontsize=18, fontweight='bold')
axs[1].grid(True, linestyle='--', alpha=0.3)

# Plot bandpass noise
axs[2].plot(time, bandpass_noise, color='r')
axs[2].set_title('Band-pass Noise', fontsize=18, fontweight='bold')
axs[2].set_xlabel(' ', fontsize=18, fontweight='bold')
axs[2].grid(True, linestyle='--', alpha=0.3)

# Adjust layout for aesthetics
plt.tight_layout()
plt.show()

# Function to compute FFT and frequency axis
def compute_fft(signal, sample_rate):
    fft_result = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), 1 / sample_rate)
    return fft_freq[:len(fft_result)//2], np.abs(fft_result[:len(fft_result)//2])

# Compute FFT for each noise signal
fft_freq_white, fft_white = compute_fft(white_noise, sample_rate)
fft_freq_notched, fft_notched = compute_fft(notched_noise, sample_rate)
fft_freq_bandpass, fft_bandpass = compute_fft(bandpass_noise, sample_rate)

# Plot the FFTs
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# Plot FFT of white noise
axs[0].plot(fft_freq_white, fft_white, color='b')
axs[0].set_title('White Noise (FFT)', fontsize=18, fontweight='bold')
axs[0].set_xlabel(' ', fontsize=18, fontweight='bold')
axs[0].set_ylabel('Magnitude', fontsize=18, fontweight='bold')
axs[0].set_xlim([0, sample_rate / 2])
axs[0].grid(True, linestyle='--', alpha=0.3)

# Plot FFT of notched noise
axs[1].plot(fft_freq_notched, fft_notched, color='g')
axs[1].set_title('Notched Noise (FFT)', fontsize=18, fontweight='bold')
axs[1].set_xlabel('Frequency (Hz)', fontsize=18, fontweight='bold')
axs[1].set_xlim([0, sample_rate / 2])
axs[1].grid(True, linestyle='--', alpha=0.3)

# Plot FFT of bandpass noise
axs[2].plot(fft_freq_bandpass, fft_bandpass, color='r')
axs[2].set_title('Band-pass Noise (FFT)', fontsize=18, fontweight='bold')
axs[2].set_xlabel(' ', fontsize=18, fontweight='bold')
axs[2].set_xlim([0, sample_rate / 2])
axs[2].grid(True, linestyle='--', alpha=0.3)

# Adjust layout for aesthetics
plt.tight_layout()
plt.show()