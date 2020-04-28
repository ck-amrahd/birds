# load all_info pickle file and generate graphs as necessary
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

info_file = 'adversarial/all_info.pickle'
epsilons = np.linspace(0, 0.1, num=10)

value_index = {'0.0': 0, '0.1': 1, '0.22': 2, '0.46': 3, '1.0': 4, '2.15': 5, '4.64': 6, '10.0': 7, '21.54': 8,
               '46.42': 9, '100.0': 10, '215.44': 11, '464.16': 12, '1000.0': 13}

# generate heatmap for given epsilon
index = 9
epsilon = epsilons[index]
heatmap_array = np.zeros((14, 14))
print(f'epsilon: {round(epsilons[index], 4)}')
print('Running...')
with open(info_file, 'rb') as read_file:
    info = pickle.load(read_file)

    normal = info['normal']
    robust_acc = normal['normal_0.0_0.0'].cpu().numpy()
    heatmap_array[:, 0] = robust_acc[index]

    lambda_equal = info['lambda_equal']
    for model_name, robust_acc in lambda_equal.items():
        robust_acc = robust_acc.cpu().numpy()
        _, lambda_1, lambda_2 = model_name.split('_')
        heatmap_array[value_index[lambda_1], value_index[lambda_2]] = robust_acc[index]

    lambda_1_zero = info['lambda_1_zero']
    for model_name, robust_acc in lambda_1_zero.items():
        robust_acc = robust_acc.cpu().numpy()
        _, lambda_1, lambda_2 = model_name.split('_')
        heatmap_array[value_index[lambda_1], value_index[lambda_2]] = robust_acc[index]

    lambda_vary = info['lambda_vary']
    for model_name, robust_acc in lambda_vary.items():
        robust_acc = robust_acc.cpu().numpy()
        _, lambda_1, lambda_2 = model_name.split('_')
        heatmap_array[value_index[lambda_1], value_index[lambda_2]] = robust_acc[index]

ax = sns.heatmap(heatmap_array, linewidths=0.5)
plt.show()
