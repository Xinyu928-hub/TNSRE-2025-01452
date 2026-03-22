from scipy.signal import butter, filtfilt
import numpy as np

def notch_filter(bp_filtered_data, fs, f0, Q):
    """
    Apply notch filter to remove specific frequency component from the signal.

    Parameters:
    - bp_filtered_data: numpy array of shape (1, 6000)
      The bandpass filtered signal.
    - fs: float
      Sampling frequency of the signal.
    - f0: float
      Frequency to be removed (notch frequency).
    - Q: float
      Quality factor of the notch filter.

    Returns:
    - filtered_data: numpy array of shape (1, 6000)
      The signal after notch filtering.
    """
    # Design notch filter
    nyquist = 0.5 * fs
    f0_norm = f0 / nyquist
    b, a = butter(2, [f0_norm - 1/(2*Q), f0_norm + 1/(2*Q)], btype='bandstop')

    # Apply notch filter using filtfilt to avoid phase shift
    filtered_data = filtfilt(b, a, bp_filtered_data)

    return filtered_data

def filter_2sIIR(signal, cutoff_freq, sampling_rate, order, filter_type='low'):
    """
    Applies a two-stage IIR filter (low/high/bandpass) to the input signal.

    Parameters:
        signal (np.ndarray): Input signal, shape (channels, samples) or (samples,)
        cutoff_freq (list or float): Cutoff frequency/frequencies (Hz)
        sampling_rate (float): Sampling frequency (Hz)
        order (int): Filter order
        filter_type (str): 'low', 'high', or 'bandpass'

    Returns:
        np.ndarray: Filtered signal with same shape as input
    """

    # Ensure signal is 2D
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    # Ensure cutoff_freq is a list
    if isinstance(cutoff_freq, (int, float)):
        cutoff_freq = [cutoff_freq]

    if filter_type == 'bandpass' and len(cutoff_freq) != 2:
        raise ValueError("For 'bandpass' filter, cutoff_freq must be a list of two values.")

    if filter_type != 'bandpass' and len(cutoff_freq) != 1:
        raise ValueError("cutoff_freq must be a single value for 'low' or 'high' filters.")

    if max(cutoff_freq) >= sampling_rate / 2:
        raise ValueError("Cutoff frequency must be less than Nyquist frequency (fs/2).")

    filtered_signal = np.zeros_like(signal)

    if filter_type == 'bandpass':
        # First apply high-pass
        b, a = butter(order, cutoff_freq[0] / (sampling_rate / 2), btype='high')
        filtered_signal = filtfilt(b, a, signal, axis=1)

        # Then apply low-pass
        b, a = butter(order, cutoff_freq[1] / (sampling_rate / 2), btype='low')
        filtered_signal = filtfilt(b, a, filtered_signal, axis=1)
    else:
        b, a = butter(order, cutoff_freq[0] / (sampling_rate / 2), btype=filter_type)
        filtered_signal = filtfilt(b, a, signal, axis=1)

    return filtered_signal


def epoch(signal, new_sampling_rate):
    """
    Segments input signal into epochs.

    Parameters:
        signal (np.ndarray): Input data, shape (channels, samples)
        new_sampling_rate (float): Sampling frequency (Hz)

    Returns:
        tuple: (epoch_data, num_epochs, epoch_length)
            - epoch_data: shape (channels, epoch_length, num_epochs)
    """
    channels, total_samples = signal.shape
    window_duration = 2     # seconds
    total_duration = 12     # seconds

    # Calculate the number of data points in a single epoch
    epoch_length = int(round(window_duration * new_sampling_rate))

    # Since there is no overlap, the number of epochs is simply total / window
    num_epochs = int(np.floor(total_duration / window_duration))

    # Initialize the output array
    epoch_data = np.zeros((channels, epoch_length, num_epochs))

    # Segment the data
    for ch in range(channels):
        for ep in range(num_epochs):
            start = ep * epoch_length
            end = start + epoch_length
            epoch_data[ch, :, ep] = signal[ch, start:end]

    return epoch_data, num_epochs, epoch_length


def CCA(signal1, signal2):
    """
    Performs Canonical Correlation Analysis (CCA) between two signals.

    Parameters:
        signal1 (np.ndarray): shape (channels_x, samples)
        signal2 (np.ndarray): shape (channels_y, samples)

    Returns:
        tuple: (w_x, w_y) - canonical weights for signal1 and signal2
    """
    X = signal1
    Y = signal2
    T = Y.shape[1]

    mean_x = np.mean(X, axis=1)
    mean_y = np.mean(Y, axis=1)

    # Compute covariance matrices
    S11 = np.cov(X)
    S22 = np.cov(Y)
    S12 = np.cov(X, Y)[:X.shape[0], X.shape[0]:]
    S21 = S12.T

    # Solve the generalized eigenvalue problem
    eigvals_x, eigvecs_x = np.linalg.eig(np.linalg.inv(S11) @ S12 @ np.linalg.inv(S22) @ S21)
    eigvals_y, eigvecs_y = np.linalg.eig(np.linalg.inv(S22) @ S21 @ np.linalg.inv(S11) @ S12)

    # Extract the canonical weight vectors corresponding to the largest eigenvalue
    max_index_x = np.argmax(eigvals_x)
    max_index_y = np.argmax(eigvals_y)

    w_x = eigvecs_x[:, max_index_x]
    w_y = eigvecs_y[:, max_index_y]

    return w_x, w_y

