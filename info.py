# load all_info pickle file and generate graphs as necessary
import pickle
import numpy as np
import matplotlib.pyplot as plt

info_file = 'adversarial/all_info.pickle'
epsilons = np.linspace(0, 0.1, num=10)

with open(info_file, 'rb') as read_file:
    info = pickle.load(read_file)

    normal = info['normal']
    best_normal = normal['normal_0.0_0.0'].cpu().numpy()

    blackout = info['blackout']
    best_blackout = blackout['blackout_0.0_0.0'].cpu().numpy()

    best_lambda_equal = []
    lambda_equal = info['lambda_equal']
    for model_name, robust_acc in lambda_equal.items():
        best_lambda_equal.append(robust_acc.cpu().numpy())

    best_lambda_equal = np.array(best_lambda_equal)
    best_lambda_equal = np.max(best_lambda_equal, axis=0)

    best_lambda_1_zero = []
    lambda_1_zero = info['lambda_1_zero']
    for model_name, robust_acc in lambda_1_zero.items():
        best_lambda_1_zero.append(robust_acc.cpu().numpy())

    best_lambda_1_zero = np.array(best_lambda_1_zero)
    best_lambda_1_zero = np.max(best_lambda_1_zero, axis=0)

    best_lambda_vary = []
    lambda_vary = info['lambda_vary']
    for model_name, robust_acc in lambda_vary.items():
        best_lambda_vary.append(robust_acc.cpu().numpy())

    best_lambda_vary = np.array(best_lambda_vary)
    best_lambda_vary = np.max(best_lambda_vary, axis=0)

    plt.plot(epsilons, best_normal, label='normal')
    plt.plot(epsilons, best_blackout, label='blackout')
    plt.plot(epsilons, best_lambda_1_zero, label='lambda_1_zero')
    plt.plot(epsilons, best_lambda_equal, label='lambda_equal')
    plt.plot(epsilons, best_lambda_vary, label='lambda_vary')
    plt.legend()
    plt.show()
