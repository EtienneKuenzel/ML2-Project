import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


import numpy as np
import matplotlib.pyplot as plt


def plot_average_ttp(file_paths, window_size=200, selected_indices=[0]):
    all_ttp_values = []
    labels = ['Relu', 'Relu+down', 'Relu+down+swap', 'Relu+down+decrease', 'Relu+down+decrease+swap', "Relu+down+lockedConv", "Relu+down+lockedConv500"]
    colors = ['r', 'b', 'g', 'brown', 'pink', "purple", "r"]

    for file_path in file_paths:
        data = load_data(file_path)
        ttp_values = np.array([data['ttp'][i].numpy() for i in selected_indices])
        all_ttp_values.append(ttp_values)

    plt.figure(figsize=(12, 5))
    for ttp_values, label, color in zip(all_ttp_values, labels, colors):
        ttp_values_mean = np.mean(np.vstack(ttp_values), axis=0)
        ttp_values_smoothed = np.convolve(ttp_values_mean, np.ones(window_size) / window_size, mode='valid')
        plt.plot(ttp_values_smoothed, linestyle='-', color=color, label=label)

    plt.xlabel('Task')
    plt.ylabel('Time to Reach 0.75 Performance')
    plt.legend()
    plt.grid(True)
    plt.show()


def compute_average_ttp(data, selected_indices, num_segments=8, segment_size=10):
    ttp_regain_values = np.array([data['last100_ttp'][0].numpy()])
    i = 0
    a = []
    for x in ttp_regain_values[0]:
        if x[0] < 10:
            a.append(i)
        i+=1
    print(a)
    all_averages = [np.mean([ttp_regain_values[0][x]], axis=0) for x in a]
    return np.mean(all_averages, axis=0)



def plot_average_ttpregain(file_paths):
    plt.figure(figsize=(12, 5))
    labels = ["Relu", "Relu+down", "Relu+down+swap", "Relu+down+decrease", "Relu+down+decrease+swap","a","b"]
    colors = ['r', 'b', 'g', 'brown', 'pink', "purple", "r"]

    selected_indices = [0]  # Indices to use

    for file_path, label, color in zip(file_paths, labels, colors):
        data = load_data(file_path)
        final_average = compute_average_ttp(data, selected_indices)
        plt.plot(final_average, linestyle='-', color=color, label=label)

    plt.ylabel('TTP Regain Values')
    plt.legend()
    plt.grid(True)
    plt.show()


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
    labels = ["Relu","Relu+down","Relu+down+swap", "Relu+down+decrease", "Relu+down+decrease+swap"]
    colors = ["red","blue","green", "brown", "pink"]
    for activation, label, color in zip(activations, labels, colors):
        frames =[]
        for i in range(40):#20
            #i*=50
            plt.figure(figsize=(8, 6))  # Optional: Adjust figure size for better clarity
            data = activation[0][i, 0, 0].flatten()
            mean_value = np.mean(data.cpu().numpy())  # Convert tensor to NumPy before computing mean
            sns.kdeplot(data, fill=True, alpha=0.2, label=label, color="blue")  # KDE plot
            plt.axvline(mean_value, linestyle="dashed", color="red", alpha=0.8, linewidth=2, label=f"{label} Mean")  # Mean line
            print(data)
            plt.xlim(-20, 20)
            plt.ylim(0, 0.5)
            plt.grid(True)
            plt.legend()
            plt.title("Task : " + str(i))
            filename = f"frame_{i}.png"
            plt.savefig(filename)
            frames.append(filename)
            plt.clf()
            # Create a GIF for activations
        with imageio.get_writer(label + ".gif", mode="I", fps=3) as writer:
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
    plt.ylabel('Time to Reach 0.75 Performance')
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    file_paths = ["outputrelu.pkl", "outputreludown.pkl","outputreludownswap.pkl","outputreludowndecrease.pkl","outputreludowndecreaseswap.pkl","outputreludowndecreaselock.pkl","outputreludowndecreaselock2.pkl"]  # Add both files
    plot_average_ttp(file_paths)
    #test_performance(file_paths)
    #timepertask(file_paths)
    plot_average_ttpregain(file_paths)
    activation(file_paths)
