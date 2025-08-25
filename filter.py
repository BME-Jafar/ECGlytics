import numpy as np
from scipy.signal import butter
from scipy.signal import sosfiltfilt

def isoline_correction(signal, number_bins=None):
    """
    Removes the DC offset (isoline) from an ECG signal using histogram analysis.

    Parameters:
        signal (ndarray): Input signal, shape (L,) or (L, NCH) or (NCH, L)
        number_bins (int, optional): Number of bins to use for histogram analysis.
                                     Defaults to min(2^10, signal length).

    Returns:
        filtered_signal (ndarray): Signal with isoline (DC offset) removed
        offset (ndarray): Estimated DC offset for each channel
        frequency_matrix (ndarray): Histogram frequencies per channel
        bins_matrix (ndarray): Histogram bin centers per channel
    """
    # Ensure signal is 2D with shape (L, NCH)
    signal = np.atleast_2d(signal)
    if signal.shape[0] < signal.shape[1]:
        signal = signal.T

    L, NCH = signal.shape
    if number_bins is None:
        number_bins = min(2**10, L)  # Default number of histogram bins

    filteredsignal = np.zeros_like(signal)
    offset = np.zeros(NCH)
    frequency_matrix = np.zeros((number_bins, NCH))  # Store histogram frequencies
    bins_matrix = np.zeros_like(frequency_matrix)    # Store histogram bin centers

    for i in range(NCH):
        # Build histogram of signal values
        freq, bins = np.histogram(signal[:, i], bins=number_bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        pos = np.argmax(freq)              # Find the most common value (mode)
        offset[i] = bin_centers[pos]       # Use it as offset
        filteredsignal[:, i] = signal[:, i] - offset[i]

        frequency_matrix[:, i] = freq
        bins_matrix[:, i] = bin_centers

    return filteredsignal.squeeze(), offset.squeeze(), frequency_matrix, bins_matrix

def ecg_low_filter(signal, samplerate, lowpass_frequency):
    """
    Applies a 3rd order low-pass Butterworth filter to the ECG signal to remove
    high-frequency noise.

    Parameters:
        signal (ndarray): Input ECG signal, shape (L,) or (L, NCH) or (NCH,L)
        samplerate (float): Sampling rate in Hz
        lowpass_frequency (float): Cutoff frequency of low-pass filter in Hz

    Returns:
        filtered (ndarray): Low-pass filtered ECG signal, same shape as input
    """
    transposeFlag = 0
    signal = np.atleast_2d(signal)

    if signal.shape[0] < signal.shape[1]: #Shape the signal as L,NCH
        signal = signal.T
        transposeFlag = 1

    original_dtype = signal.dtype
    signal = signal.astype(np.float64)
    L, NCH = signal.shape

    # Extend signal to reduce edge artifacts
    l = int(round(samplerate * 10))
    extended_signal = np.zeros((L + 2 * l, NCH))
    for i in range(NCH):
        extended_signal[:, i] = np.pad(signal[:, i], (l, l), mode='symmetric')

    # Validate lowpass frequency
    if lowpass_frequency > samplerate / 2:
        print("Warning: Low-pass frequency above Nyquist. Using Nyquist-1 instead.")
        lowpass_frequency = np.floor(samplerate / 2 - 1)

    # Design Butterworth low-pass filter
    order = 3
    sos = butter(order, 2 * lowpass_frequency / samplerate, btype='low', output='sos')

    # Apply filter
    for i in range(NCH):
        extended_signal[:, i] = sosfiltfilt(sos, extended_signal[:, i])

    # Trim the extended part
    filtered = extended_signal[l:-l, :]

    # Apply isoline correction
    def isoline_correction_inner(sig):
        corrected, *_ = isoline_correction(sig)
        return corrected

    filtered = np.column_stack([isoline_correction_inner(filtered[:, i]) for i in range(NCH)])

    if original_dtype != np.float32:
        filtered = filtered.astype(original_dtype)
    
    if transposeFlag:
        filtered = filtered.T

    return filtered.squeeze()

def ecg_high_filter(signal, samplerate, highpass_frequency):
    """
    Applies a 3rd order high-pass Butterworth filter to the ECG signal to remove
    low-frequency components

    Parameters:
        signal (ndarray): Input ECG signal, shape (L,) or (L, NCH) or (NCH,L)
        samplerate (float): Sampling rate in Hz
        highpass_frequency (float): Cutoff frequency of high-pass filter in Hz

    Returns:
        filtered (ndarray): High-pass filtered ECG signal, same shape as input
    """
    transposeFlag = 0
    signal = np.atleast_2d(signal)

    if signal.shape[0] < signal.shape[1]: #Shape the signal as L,NCH
        signal = signal.T
        transposeFlag = 1

    original_dtype = signal.dtype
    signal = signal.astype(np.float64)

    L, NCH = signal.shape

    # Extend signal to reduce edge artifacts during filtering
    l = int(round(samplerate * 10))
    extended_signal = np.zeros((L + 2 * l, NCH))
    for i in range(NCH):
        extended_signal[:, i] = np.pad(signal[:, i], (l, l), mode='symmetric')

    # Validate highpass frequency (must be below Nyquist)
    if highpass_frequency > samplerate / 2:
        print("Warning: High-pass frequency above Nyquist. Using Nyquist-1 instead.")
        highpass_frequency = np.floor(samplerate / 2 - 1)

    # Design Butterworth high-pass filter
    order = 3
    sos = butter(order, 2 * highpass_frequency / samplerate, btype='high', output='sos')

    # Apply filter
    for j in range(NCH):
        extended_signal[:, j] = sosfiltfilt(sos, extended_signal[:, j])

    # Remove extension
    filtered = extended_signal[l:-l, :]

    # Apply isoline correction to each channel
    def isoline_correction_inner(sig):
        corrected, offset, *_ = isoline_correction(sig)
        return corrected

    filtered = np.column_stack([isoline_correction_inner(filtered[:, i]) for i in range(NCH)])

    if original_dtype != np.float32:
        filtered = filtered.astype(original_dtype)
    
    if transposeFlag:
        filtered = filtered.T

    return filtered.squeeze()

