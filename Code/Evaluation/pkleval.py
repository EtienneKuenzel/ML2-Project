import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def plot_average_ttp(file_paths):
    window_size = 200  # Adjust as needed
    selected_indices = [1, 2, 3]  # Indices to use

    all_ttp_values1 = []
    all_ttp_values2 = []
    all_ttp_values3 = []

    data = load_data(file_paths[0])
    ttp_values = np.array([data['ttp'][i].numpy() for i in selected_indices])
    all_ttp_values1.append(ttp_values)
    selected_indices = [0, 1, 2]  # Indices to use
    data = load_data(file_paths[1])
    ttp_values = np.array([data['ttp'][i].numpy() for i in selected_indices])
    all_ttp_values2.append(ttp_values)

    data = load_data(file_paths[2])
    ttp_values = np.array([data['ttp'][i].numpy() for i in selected_indices])
    all_ttp_values3.append(ttp_values)


    # Stack all collected values and compute the overall mean
    ttp_values_smoothed1 = np.convolve(np.mean(np.vstack(all_ttp_values1), axis=0), np.ones(window_size) / window_size, mode='valid')
    ttp_values_smoothed2 = np.convolve(np.mean(np.vstack(all_ttp_values2), axis=0), np.ones(window_size) / window_size, mode='valid')
    ttp_values_smoothed3 = np.convolve(np.mean(np.vstack(all_ttp_values3), axis=0), np.ones(window_size) / window_size, mode='valid')

    plt.figure(figsize=(12, 5))
    plt.plot(ttp_values_smoothed1, linestyle='-', color='r', label='Relu')
    plt.plot(ttp_values_smoothed2, linestyle='-', color='b', label='Relu+down')
    plt.plot(ttp_values_smoothed3, linestyle='-', color='g', label='Relu+down+swap')
    plt.xlabel('Index')
    plt.ylabel('TTP Values')
    plt.title('Averaged Plot of TTP Values with Smoothing (Combined Files)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_average_ttpregain(file_paths):
    plt.figure(figsize=(12, 5))

    selected_indices = [1,2, 3]  # Indices to use
    data = load_data(file_paths[0])
    ttp_regain_values = np.array([data['last100_ttp'][i].numpy() for i in selected_indices])
    all_averages = []
    for x in range(10):
        avg = np.mean([ttp_regain_values[0][x * 50],ttp_regain_values[1][x * 50],ttp_regain_values[2][x * 50]], axis=0)
        all_averages.append(avg)
    # Compute the final average over all collected averages
    final_average = np.mean(all_averages, axis=0)
    plt.plot(final_average, linestyle='-', color='r', label="Relu")


    selected_indices = [0, 1, 2]  # Indices to use
    data = load_data(file_paths[1])
    ttp_regain_values = np.array([data['last100_ttp'][i].numpy() for i in selected_indices])
    all_averages = []
    for x in range(5):
        avg = np.mean([ttp_regain_values[0][x * 50], ttp_regain_values[1][x * 50], ttp_regain_values[2][x * 50]], axis=0)
        all_averages.append(avg)
    # Compute the final average over all collected averages
    final_average = np.mean(all_averages, axis=0)
    plt.plot(final_average, linestyle='-', color='b', label="Relu+down")

    selected_indices = [0, 1, 2]  # Indices to use
    data = load_data(file_paths[2])
    ttp_regain_values = np.array([data['last100_ttp'][i].numpy() for i in selected_indices])
    all_averages = []
    for x in range(5):
        avg = np.mean([ttp_regain_values[0][x * 50], ttp_regain_values[1][x * 50], ttp_regain_values[2][x * 50]],
                      axis=0)
        all_averages.append(avg)
    # Compute the final average over all collected averages
    final_average = np.mean(all_averages, axis=0)
    plt.plot(final_average, linestyle='-', color='g', label="Relu+down+swap")

    plt.ylabel('TTP Regain Values')
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    file_paths = ["outputrelu.pkl", "outputreludown.pkl","outputreludownswap.pkl"]  # Add both files
    plot_average_ttp(file_paths)
    plot_average_ttpregain(file_paths)
