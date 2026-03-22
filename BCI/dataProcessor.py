import numpy as np
from BCI.ReceiveData import LSLDataCollector
from utils.def_function import CCA, notch_filter, filter_2sIIR

class EEGDataProcessor:
    def __init__(self, freqs=[13, 12, 11, 10, 9], bp_freq = [4,39], notch_freq = 50, order = 6, Q = 30, sample_rate = 500):
        """
        EEG Data Processor for SSVEP recognition.

        This class can be used in offline or online testing mode.

        Args:
            sample_rate
            freqs (list): List of target frequencies.
            notch_freq (int): Bandpass cutoff frequencies [low, high].
        """
        self.sample_rate = sample_rate
        self.notch_freq = notch_freq
        self.Q = Q
        self.order = order
        self.bp_freq = bp_freq
        self.freqs = freqs
        self.model = np.load('your path\model_data.npy', allow_pickle=True).item()

    def predict_online(self, trial):
        """
        Online testing: input one trial of EEG data (channels x samples) and predict frequency.
        """
        if self.model is None:
            raise ValueError("No model loaded. Use 'from_online_stream()'.")
        coeffs = self._compute_coefficients(trial)
        best_index = np.argmax(coeffs)
        predicted_freq = self.freqs[best_index]
        #print(f"[Online Prediction] Predicted Frequency: {predicted_freq} Hz")
        return predicted_freq


    def _compute_coefficients(self, test_data):
        """
        Compute frequency correlation coefficients for one EEG trial.
        """
        filtered_data_list = []
        for i in range(test_data.shape[0]):
            origin_data = np.transpose(test_data[i, :])
            bp_filtered_data = filter_2sIIR(origin_data, self.bp_freq, self.sample_rate, self.order, 'bandpass')
            nf_filtered_data = notch_filter(bp_filtered_data, self.sample_rate, self.notch_freq, self.Q)
            filtered_data_list.append(nf_filtered_data)
        test_data = np.concatenate(filtered_data_list, axis=0)

        num_freqs = len(self.freqs)
        trial_coefficients = np.zeros(num_freqs)

        for i, freq in enumerate(self.freqs):
            # ρ1: Test vs Reference
            w_test1, w_ref1 = CCA(self.model['Reference'][:, :, i], test_data)
            proj_test1 = w_test1 @ test_data
            proj_ref1 = w_ref1 @ self.model['Reference'][:, :, i]
            rho1 = abs(np.corrcoef(proj_test1, proj_ref1)[0, 1])

            # ρ2: Test vs Template
            w_temp, _ = CCA(self.model['Template'][:, :, i], test_data)
            proj_test2 = w_temp @ test_data
            proj_temp2 = w_temp @ self.model['Template'][:, :, i]
            rho2 = np.corrcoef(proj_test2, proj_temp2)[0, 1]

            # ρ3: Test vs Reference again
            w_ref, _ = CCA(self.model['Reference'][:, :, i], test_data)
            proj_test3 = w_ref @ test_data
            proj_temp3 = w_ref @ self.model['Template'][:, :, i]
            rho3 = np.corrcoef(proj_test3, proj_temp3)[0, 1]

            # ρ4: Template vs Reference
            w_temp_ref, _ = CCA(self.model['Template'][:, :, i], self.model['Reference'][:, :, i])
            proj_test4 = w_temp_ref @ test_data
            proj_temp4 = w_temp_ref @ self.model['Template'][:, :, i]
            rho4 = np.corrcoef(proj_test4, proj_temp4)[0, 1]

            correlations = np.array([rho1, rho2, rho3, rho4])
            signs = np.sign(correlations)
            trial_coefficients[i] = abs(np.sum(signs * correlations**2))

        return trial_coefficients

if __name__ == '__main__':
    model_path = 'your path\model_data.npy'
    Collector = LSLDataCollector()
    Collector.initialize_inlet()
    Collector.collect_data()
    Processor = EEGDataProcessor()
    label = Processor.predict_online(Collector.buffer)
    #print(f"Predicted label: {label}")
