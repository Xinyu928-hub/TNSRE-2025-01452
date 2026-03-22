

import numpy as np
import scipy.io
import os
from utils.def_function import epoch, CCA, filter_2sIIR,notch_filter

if __name__ == '__main__':
    # ---------------------------------------------------------
    # 1. Constants and Configuration
    # ---------------------------------------------------------
    SAMPLE_RATE = 500
    N_HARMONIC = 4
    FREQS = [13, 12, 11, 10, 9]
    ORDER = 7
    CUTOFF_FREQ = [4, 64]
    NORTH_FREQ = 50
    Q = 30  # Quality factor of the notch filter.
    N_CHANNELS_KEEP = 8  # Number of channels to keep in the end
    TW = 2  # Time window size for ITR calculation

    # Define the file paths
    data_paths = {
        'f1data': r'your path\f1data.mat',
        'f2data': r'your path\f2data.mat',
        'f3data': r'your path\f3data.mat',
        'f4data': r'your path\f4data.mat',
        'f5data': r'your path\f5data.mat'
    }
    # ---------------------------------------------------------
    # 2. Data Loading and Preprocessing
    # ---------------------------------------------------------
    processed_data = {}

    for key, path in data_paths.items():
        data = scipy.io.loadmat(path)[key]
        epochs_per_block = []

        # Iterate over Blocks (data.shape[2])
        for i in range(data.shape[2]):
            filtered_channels = []

            # Iterate over Channels (data.shape[1])
            for j in range(data.shape[1]):
                origin_data = data[:, j, i].T
                bp_filtered_data = filter_2sIIR(origin_data, CUTOFF_FREQ, SAMPLE_RATE, ORDER, 'bandpass')
                nf_filtered_data = notch_filter(bp_filtered_data, SAMPLE_RATE, NORTH_FREQ, Q)
                filtered_channels.append(nf_filtered_data)

            # Concatenate filtered single-channel data and segment into Epochs
            filtered_concat = np.concatenate(filtered_channels, axis=0)
            epoch_data, _, _ = epoch(filtered_concat, SAMPLE_RATE)

            # Adjust dimension order
            epoch_data = np.transpose(epoch_data, (2, 1, 0))
            epochs_per_block.append(epoch_data)

        # Concatenate all Blocks for the current subject/frequency
        combined_epochs = np.concatenate(epochs_per_block, axis=0)
        combined_epochs = np.transpose(combined_epochs, (2, 1, 0))

        # Keep only the first N_CHANNELS_KEEP channels
        processed_data[key] = combined_epochs[:N_CHANNELS_KEEP, :, :]


    # ---------------------------------------------------------
    # 3. Organize multidimensional data arrays (eeg & eeg2) and labels
    # ---------------------------------------------------------
    # eeg shape: (N_channel, N_point, N_block, N_target)
    eeg = np.stack([processed_data[key] for key in data_paths.keys()], axis=-1)

    N_channel, N_point, N_block, N_target = eeg.shape

    # eeg2 shape: (N_channel, N_point, N_target * N_block) -> Flatten Trials
    eeg2 = np.concatenate([eeg[:, :, :, i] for i in range(N_target)], axis=2)
    NumTrial = eeg2.shape[2]

    # Generate true labels (allocated by the number of Trials per Target)
    trials_per_target = NumTrial // N_target
    labels = np.repeat(FREQS, trials_per_target)

    # ---------------------------------------------------------
    # 4. Build SSVEP reference template and standard signal model
    # ---------------------------------------------------------
    model = {
        'Template': np.zeros((N_channel, N_point, N_target)),
        'Reference': np.zeros((2 * N_HARMONIC, N_point, N_target))
    }

    t = np.arange(1, N_point + 1) / SAMPLE_RATE

    for targ_i, freq in enumerate(FREQS):
        # Template: Average across the Block dimension
        model['Template'][:, :, targ_i] = np.mean(eeg[:, :, :, targ_i], axis=2)

        # Reference: Generate sine and cosine reference signals for the fundamental frequency and its harmonics
        y_ref = []
        for har_i in range(1, N_HARMONIC + 1):
            y_ref.append(np.sin(2 * np.pi * freq * har_i * t))
            y_ref.append(np.cos(2 * np.pi * freq * har_i * t))
        model['Reference'][:, :, targ_i] = np.array(y_ref)

    # ---------------------------------------------------------
    # 5. CCA Correlation Calculation and Classification Prediction
    # ---------------------------------------------------------
    all_coeffs_history = []
    output_labels = np.zeros(NumTrial, dtype=int)

    for trial_i in range(NumTrial):
        test_data = eeg2[:, :, trial_i]
        trial_coeffs = np.zeros(len(FREQS))

        for targ_j in range(len(FREQS)):
            ref = model['Reference'][:, :, targ_j]
            temp = model['Template'][:, :, targ_j]

            # 1. Standard CCA
            w_test, w_ref = CCA(ref, test_data)
            rho1 = abs(np.corrcoef(w_test @ test_data, w_ref @ ref)[0, 1])

            # 2. IT-CCA
            w_test, _ = CCA(temp, test_data)
            rho2 = np.corrcoef(w_test @ test_data, w_test @ temp)[0, 1]

            # 3. Cross-CCA 1
            w_test, _ = CCA(ref, test_data)
            rho3 = np.corrcoef(w_test @ test_data, w_test @ temp)[0, 1]

            # 4. Cross-CCA 2
            w_test, _ = CCA(temp, ref)
            rho4 = np.corrcoef(w_test @ test_data, w_test @ temp)[0, 1]

            # Fuse the 4 correlation coefficients
            rhos = np.array([rho1, rho2, rho3, rho4])
            trial_coeffs[targ_j] = abs(np.sum(np.sign(rhos) * (rhos ** 2)))

        all_coeffs_history.append(trial_coeffs)
        output_labels[trial_i] = FREQS[np.argmax(trial_coeffs)]

    # Corresponds to the allcoeff history matrix in the original code
    allcoeff = np.array(all_coeffs_history)

    # ---------------------------------------------------------
    # 6. Accuracy Evaluation and ITR Calculation
    # ---------------------------------------------------------
    trueNum = np.sum(output_labels == labels)
    acc = trueNum / len(labels)

    # Save model
    np.save('model_data.npy', model)

    # Terminal output (format matches the original)
    print(f'\nThe number of correct predictions: {trueNum}/{NumTrial}')
    print(f'Cross-validation average accuracy: {acc:.4f}')

    Nf = len(FREQS)
    if acc == 1.0:
        itr = (np.log2(Nf) + acc * np.log2(acc)) * (60 / TW)
    elif acc < 1 / Nf:
        itr = 0.0
    else:
        itr = (np.log2(Nf) + acc * np.log2(acc) + (1 - acc) * np.log2((1 - acc) / (Nf - 1))) * (60 / TW)

    print(f'The ITR is: {itr:.4f} bmp')