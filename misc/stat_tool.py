import numpy as np

def rmSignal(data, signal):
    data_a = data - np.mean(data)
    signal_a = signal - np.mean(signal)

    return data_a -( np.inner(data_a, signal_a) / np.inner(signal_a, signal_a)) * signal_a

