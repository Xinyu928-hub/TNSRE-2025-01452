"""
Example program to show how to read a multi-channel time series from LSL.
"""

import numpy as np
from pylsl import StreamInlet, resolve_stream

class LSLDataCollector:
    def __init__(self, window_length=2.5, sampling_rate=500, channel_count=8):
        """
        Initialize the LSL data collector.

        Args:
            window_length (int): Duration of the data window in seconds.
            sampling_rate (int): Sampling frequency in Hz.
            channel_count (int): Number of EEG channels.
        """
        self.window_length = window_length
        self.fs = sampling_rate
        self.channel_count = channel_count
        self.num_points = int(self.window_length * self.fs)

        # Pre-allocate data buffer [channels, samples]
        self.buffer = np.zeros((self.channel_count, self.num_points))

        self.inlet = None

    def initialize_inlet(self, stream_type='EEG'):
        """
        Resolve and initialize the LSL stream inlet.
        """
        print(f"Resolving LSL stream of type '{stream_type}'...")
        streams = resolve_stream('type', stream_type)
        if not streams:
            raise RuntimeError(f"No LSL stream of type '{stream_type}' found.")
        self.inlet = StreamInlet(streams[0])
        print("LSL stream connected.")

    def collect_data(self, num_windows=1):
        """
        Collect EEG data for a specified number of time windows.

        Args:
            num_windows (int): Number of data windows to collect.
        """
        for window_index in range(num_windows):
            #print(f"\nCollecting window {window_index + 1}/{num_windows}...")

            for i in range(self.num_points):
                sample, timestamp = self.inlet.pull_sample()
                if timestamp is not None:
                    if len(sample) < self.channel_count:
                        raise ValueError("Received sample has fewer channels than expected.")
                    self.buffer[:, i] = sample[:self.channel_count]

            # Transpose to shape (samples, channels)
            collected_window = self.buffer
            #print(f"Collected data shape: {collected_window.shape}")

if __name__ == '__main__':
    print("Looking for EEG stream...")
    collector = LSLDataCollector()
    collector.initialize_inlet()
    collector.collect_data()