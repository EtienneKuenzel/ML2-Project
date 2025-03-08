import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def plot_ttp(data):
    # Ensure the 'ttp' key exists in the data
    for x in range(3):
        window_size = 128  # Adjust this based on your needs
        ttp_values = data['ttp'][x + 1].numpy()
        ttp_values_smoothed = np.convolve(ttp_values, np.ones(window_size) / window_size, mode='valid')
        plt.figure(figsize=(12, 5))
        plt.plot(ttp_values_smoothed, marker='o', linestyle='-', color='r', label='Smoothed TTP Values')
        plt.xlabel('Index')
        plt.ylabel('TTP Values')
        plt.title('Plot of TTP Values with Smoothing')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    data_file = "outputrelu.pkl"
    plot_ttp(load_data(data_file))
