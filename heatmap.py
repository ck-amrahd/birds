# load all_info pickle file and generate graphs as necessary
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

info_file = 'adversarial/all_info_exp6.pickle'
epsilons = np.linspace(0, 0.2, num=10)

value_index = {'0.0': 0, '0.1': 1, '0.22': 2, '0.46': 3, '1.0': 4, '2.15': 5, '4.64': 6, '10.0': 7, '21.54': 8,
               '46.42': 9, '100.0': 10, '215.44': 11, '464.16': 12, '1000.0': 13}

# generate heatmap for given epsilon
heatmap_array = np.zeros((14, 14))
print('Running...')

for index, epsilon in enumerate(epsilons):
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

        # lambda_1_zero = info['lambda_1_zero']
        # for model_name, robust_acc in lambda_1_zero.items():
        #    robust_acc = robust_acc.cpu().numpy()
        #    _, lambda_1, lambda_2 = model_name.split('_')
        #    heatmap_array[value_index[lambda_1], value_index[lambda_2]] = robust_acc[index]

        lambda_vary = info['lambda_vary']
        for model_name, robust_acc in lambda_vary.items():
            robust_acc = robust_acc.cpu().numpy()
            _, lambda_1, lambda_2 = model_name.split('_')
            heatmap_array[value_index[lambda_1], value_index[lambda_2]] = robust_acc[index]

    plt.figure(figsize=(19.20, 10.80))
    x_axis_labels = [0.0, 0.1, 0.22, 0.46, 1.0, 2.15, 4.64, 10.0, 21.54, 46.42, 100.0, 215.44, 464.16, 1000.0]
    y_axis_labels = [0.0, 0.1, 0.22, 0.46, 1.0, 2.15, 4.64, 10.0, 21.54, 46.42, 100.0, 215.44, 464.16, 1000.0]
    ax = sns.heatmap(heatmap_array, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels, linewidths=0.5)
    ax.set(xlabel='lambda_2', ylabel='lambda_1')
    epsilon = str(round(epsilon, 4))
    plt.title('epsilon=' + epsilon)
    plt.savefig('heatmap' + '/' + 'epsilon=' + epsilon + '.png', format='png')

print('Done')
