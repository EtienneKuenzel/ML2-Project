import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import numpy as np
import matplotlib.pyplot as plt

# Define initial layer scaling values
layer_scaling = {
    "conv1": 0.75,
    "conv2": 0.8,
    "conv3": 0.85,
    "fc1": 0.9,
    "fc2": 0.95,
    "fc3": 1.0
}

# Compute layer scaling values for tasks 0 to 2000
tasks = np.arange(0, 2001)
scaling_values = {name: [scale + (1 - scale) * 1.005**-task for task in tasks] for name, scale in layer_scaling.items()}

# Plot the results
plt.figure(figsize=(10, 6))
for name, values in scaling_values.items():
    plt.plot(tasks, values, label=name)

plt.xlabel("Task")
plt.ylabel("Layer Scaling Value")
plt.title("Layer Scaling Progression from Task 0 to 2000")
plt.legend()
plt.grid(True)
plt.show()

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def plot_average_ttp(file_paths, window_size=500, selected_indices=[0]):
    all_ttp_values = []
    labels = ['Relu+down',"Relu+down+decrease 0.05","Relu+down+decrease 0.10", "Relu+down+decrease 0.15", "Relu+down(Convolutions locked)", "Relu+down(FC locked)"]
    colors = ['r', 'b', 'g', 'brown', 'pink', "purple", "r"]

    for file_path in file_paths:
        data = load_data(file_path)
        ttp_values = np.array([data['ttp'][i].numpy() for i in selected_indices])

        # Subtract the first value so each plot starts at zero
        ttp_values = ttp_values - ttp_values[:, 0][:, np.newaxis]

        all_ttp_values.append(ttp_values)

    plt.figure(figsize=(12, 5))
    for ttp_values, label, color in zip(all_ttp_values, labels, colors):
        ttp_values_mean = np.mean(np.vstack(ttp_values), axis=0)
        ttp_values_smoothed = np.convolve(ttp_values_mean, np.ones(window_size) / window_size, mode='valid')

        # Ensure first point starts at zero
        ttp_values_smoothed -= ttp_values_smoothed[0]
        if label == "Relu+down(FC locked)":
            plt.plot(ttp_values_smoothed+109, linestyle='-', color=color, label=label)
        else:
            plt.plot(ttp_values_smoothed+51, linestyle='-', color=color, label=label)



    plt.xlabel('Task')
    plt.ylabel('Epochs to Reach 85% Accuracy')
    plt.legend()
    plt.xlim(-10,1500)
    plt.ylim(0,150)
    plt.grid(True)
    plt.show()


def compute_average_ttp(data, selected_indices, num_segments=8, segment_size=10):
    ttp_regain_values = np.array([data['last100_ttp'][0].numpy()])
    i = 0
    a = []
    for x in ttp_regain_values[0]:
        if x[0] < 10 and  len(a)<20:
            a.append(i)
        i+=1
    all_averages = [np.mean([ttp_regain_values[0][x]], axis=0) for x in a]
    return np.mean(all_averages, axis=0)



import numpy as np
import matplotlib.pyplot as plt

def plot_average_ttpregain(file_paths):
    plt.figure(figsize=(12, 5))  # Increase DPI for sharpness
    labels = ["Relu", "Relu+down", "Relu+down+\nswap", "Relu+down+\ndecrease", "Relu+down+\ndecrease+swap", "a", "b"]
    colors = ['r', 'b', 'g', 'brown', 'pink', "purple", "r"]

    selected_indices = [0]  # Indices to use
    means = []
    stds = []
    valid_labels = []
    valid_colors = []

    for file_path, label, color in zip(file_paths, labels, colors):
        try:
            data = load_data(file_path)
            final_average = compute_average_ttp(data, selected_indices)
            mean_val = final_average.mean()
            std_val = final_average.std()
            print(f"{label}: Mean={mean_val}, Std={std_val}")
            means.append(mean_val/9)
            stds.append(std_val/9)
            valid_labels.append(label)
            valid_colors.append(color)
        except Exception as e:
            print(f"Skipping {label} due to error: {e}")


    if len(means) == len(valid_labels) == len(stds) == len(valid_colors):
        plt.bar(valid_labels, means, yerr=stds, capsize=8, color=valid_colors, alpha=0.85, edgecolor='black', linewidth=1.2)
        plt.ylabel('Epochs to Relearn the older tasks', fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.6, linewidth=0.7)
        plt.show()
    else:
        print("Error: Mismatch in data lengths, skipping plot.")


def timepertask(filepaths):
    for path in filepaths:
        data = load_data(path)
        print(data['time per task'])
def activation(filepaths):
    data_list = []
    for file_path in filepaths:
        with open(file_path, 'rb') as file:
            data_list.append(pickle.load(file))
    activations = [data['task_activations'] for data in data_list]
    labels = ["Relu+down (FC Locked)",]
    colors = ["red","blue","green", "brown", "pink"]
    for activation, label, color in zip(activations, labels, colors):
        print(label)
        frames =[]
        for i in range(200):#20
            #i*=50
            plt.figure(figsize=(8, 6))  # Optional: Adjust figure size for better clarity
            data = activation[0][i, 0, 0].flatten()
            mean_value = np.mean(data.cpu().numpy())  # Convert tensor to NumPy before computing mean
            sns.kdeplot(data, fill=True, alpha=0.2, label=label, color="blue")  # KDE plot
            plt.axvline(mean_value, linestyle="dashed", color="red", alpha=0.8, linewidth=2, label=f"{label} Mean")  # Mean line
            plt.xlim(-20, 20)
            plt.ylim(0, 0.5)
            plt.grid(True)
            plt.legend()
            plt.title("Task : " + str(i*10))
            filename = f"frame_{i}.png"
            plt.savefig(filename)
            frames.append(filename)
            plt.clf()
            # Create a GIF for activations
        with imageio.get_writer(label + ".gif", mode="I", fps=10) as writer:
            for frame in frames: writer.append_data(imageio.imread(frame))
def test_performance(filepaths):
    selected_indices = [0]  # Indices to use
    window_size =2
    all_ttp_values1 = []
    all_ttp_values2 = []
    all_ttp_values3 = []
    all_ttp_values4 = []
    all_ttp_values5 = []
    plt.figure(figsize=(12, 5))

    data = load_data(file_paths[0])
    for x in range(10):
        x+=980
        ttp_values = np.array([data['test_accuracies'][i][x].numpy() for i in selected_indices])
        all_ttp_values1.append(ttp_values)
        ttp_values_smoothed1 = np.convolve(np.mean(np.vstack(all_ttp_values1), axis=0), np.ones(window_size) / window_size, mode='valid')
        plt.plot(ttp_values_smoothed1, linestyle='-', linewidth=0.1, color='r', label=x)


    plt.xlabel('Task')
    plt.ylabel('Time to Reach 0.85 Performance')
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    file_paths = [
        "reludown.pkl",
        "reludown+decrease0.05.pkl",
        "reludown+decrease0.1.pkl",
        "reludown+decrease0.15.pkl",
        #"reludown+decrease+swap.pkl",
        "reludown+lock.pkl",
        "reludown+lockr.pkl"
    ]  # Add both files
    plot_average_ttp(file_paths)
    test_performance(file_paths)
    #timepertask(file_paths)
    plot_average_ttpregain(file_paths)
    activation(file_paths)
