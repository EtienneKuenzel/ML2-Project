import pickle
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
import math

# Initial layer scaling settings
"""layer_scaling = {
    "conv1.weight": 0.5,
    "conv1.bias": 0.5,
    "conv2.weight": 0.6,
    "conv2.bias": 0.6,
    "conv3.weight": 0.7,
    "conv3.bias": 0.7,
    "fc1.weight": 0.8,
    "fc1.bias": 0.8,
    "fc2.weight": 0.9,
    "fc2.bias": 0.9,
    "fc3.weight": 1.0,
    "fc3.bias": 1.0
}

# Scaling function applied to each layer
task = 1000
# Generate task values from 0 to 1000
tasks = np.arange(0, 1001)

# Calculate the layer scaling for each task value
evolving_scaling = {
    name: scale + (1 - scale) * 1.005**(-tasks)#np.exp(-tasks)
    for name, scale in layer_scaling.items()
}

# Plot the results
plt.figure(figsize=(10, 6))

for name, scaling_values in evolving_scaling.items():
    plt.plot(tasks, scaling_values, label=name)

plt.title("Layer Scaling Evolution from Task 0 to 1000")
plt.xlabel("Task")
plt.ylabel("Scaling Factor")
plt.legend()
plt.grid(True)
plt.show()"""
import pickle
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio


def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def analyze_metrics(data):
    print("--- Evaluation Metrics ---")

    # Time to performance regain (average over 50 tasks)
    ttp_last100 = data['last100_ttp']
    print(len(ttp_last100))
    for x in ttp_last100:
        print(x[0])
    avg_ttp_regain = ttp_last100.mean(dim=1).mean().item()
    print(f"Average Time to Performance Regain: {avg_ttp_regain:.2f}")

    # Time to performance (averaged over tasks)
    ttp = data['ttp']
    smoothed_ttp = np.convolve(ttp.numpy(), np.ones(1) / 1, mode='valid')  # Moving average smoothing
    print(f"Time to performance (smoothed): {smoothed_ttp.mean():.2f} ± {smoothed_ttp.std():.2f}")

    # Train and test accuracies
    train_accuracies = data['train_accuracies']
    test_accuracies = data['test_accuracies']

    print(f"Final Train Accuracy: {train_accuracies[:, -1].mean():.2f} ± {train_accuracies[:, -1].std():.2f}")
    print(f"Final Test Accuracy: {test_accuracies[:, -1].mean():.2f} ± {test_accuracies[:, -1].std():.2f}")

    # Time per task
    time_per_task = data['time per task']
    print(f"Average time per task: {time_per_task:.2f} sec")

    return train_accuracies, test_accuracies, smoothed_ttp, time_per_task, ttp_last100


def plot_accuracies(train_accuracies, test_accuracies):
    num_tasks, num_epochs = train_accuracies.shape
    plt.figure(figsize=(10, 5))
    for task in range(num_tasks):
        plt.plot(range(num_epochs), train_accuracies[task], label=f'Train Task {task}')
        plt.plot(range(num_epochs), test_accuracies[task], label=f'Test Task {task}', linestyle='dashed')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train vs Test Accuracy Over Epochs')
    plt.legend()
    plt.show()


def plot_ttp(ttp):
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(ttp)), ttp, marker='o', linestyle='-')
    plt.xlabel('Task')
    plt.ylabel('Time to Performance')
    plt.title('Time to Performance per Task (Smoothed)')
    plt.grid()
    plt.show()


def create_ttp_gif(ttp_last100, filename='ttp_last100.gif'):
    frames = []
    num_tasks = ttp_last100.shape[0]

    for task in range(0, num_tasks, 50):  # Averaging over 50 tasks
        avg_regain_time = ttp_last100[max(0, task - 49):task + 1, :10].mean(dim=0)

        plt.figure(figsize=(8, 5))
        plt.plot(range(10), avg_regain_time.numpy(), marker='o', linestyle='-')
        plt.xlabel('Last 9 Tasks')
        plt.ylabel('Average Regain Time')
        plt.title(f'Avg Time to Performance Regain - Task {task}')
        plt.grid()
        plt.ylim(0, ttp_last100.max())

        # Save frame
        plt.savefig("temp_frame.png")
        plt.show()
        plt.close()
        frames.append(imageio.imread("temp_frame.png"))

    imageio.mimsave(filename, frames, duration=0.5)
    print(f"Saved GIF: {filename}")


def main():
    data_file = "outputrelu.pkl"
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found.")
        return

    data = load_data(data_file)
    train_accuracies, test_accuracies, ttp, time_per_task, ttp_last100 = analyze_metrics(data)
    #plot_accuracies(train_accuracies, test_accuracies)
    plot_ttp(ttp)
    create_ttp_gif(ttp_last100)


if __name__ == "__main__":
    main()

