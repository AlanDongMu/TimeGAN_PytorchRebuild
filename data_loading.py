import numpy as np
import os

def MinMaxScaler(data):
    """Min Max normalizer.
    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def sine_data_generation(no, seq_len, dim):
    """Sine data generation.
    Args:
      - no: the number of samples
      - seq_len: sequence length of the time-series
      - dim: feature dimensions

    Returns:
      - data: generated data
    """
    # Initialize the output
    data = list()

    # Generate sine data
    for i in range(no):
        # Initialize each time-series
        temp = list()
        # For each feature
        for k in range(dim):
            # Randomly drawn frequency and phase
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))
        # Normalize to [0,1]
        temp = (temp + 1) * 0.5
        # Stack the generated data
        data.append(temp)

    return data


def real_data_loading(data_dir, data_name, seq_len):
    """Load and preprocess real-world datasets.
    Args:
      - data_name: stock or energy
      - seq_len: sequence length
    Returns:
      - data: preprocessed data.
    """

    assert data_name in ['stock', 'energy']

    ori_data = []
    if data_name == 'stock':
        ori_data = np.loadtxt(os.path.join(data_dir, 'stock_data.csv'), delimiter=",", skiprows=1)
    elif data_name == 'energy':
        ori_data = np.loadtxt(os.path.join(data_dir, 'energy_data.csv'), delimiter=",", skiprows=1)

    # Flip the data to make chronological data
    ori_data = ori_data[::-1]
    # Normalize the data
    ori_data = MinMaxScaler(ori_data)

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    return data
