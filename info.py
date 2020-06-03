# load all_info pickle file and generate graphs as necessary
import pickle
import numpy as np
import matplotlib.pyplot as plt

info_file = 'adversarial/all_info_exp5.pickle'
num_epsilons = 20
epsilons = np.linspace(0, 0.2, num=num_epsilons)

best_models_path = 'adversarial/best_models_exp5.pickle'
best_models = {}

with open(info_file, 'rb') as read_file:
    info = pickle.load(read_file)

    normal = info['normal']
    best_normal = normal['normal_0.0_0.0'].cpu().numpy()
    best_models_normal = ['normal_0.0_0.0'] * num_epsilons
    best_models['normal'] = best_models_normal

    blackout = info['blackout']
    best_blackout = blackout['blackout_0.0_0.0'].cpu().numpy()
    best_models_blackout = ['blackout_0.0_0.0'] * num_epsilons
    best_models['blackout'] = best_models_blackout

    best_lambda_equal = [float(0)] * num_epsilons
    best_models_lambda_equal = ['None'] * num_epsilons

    lambda_equal = info['lambda_equal']
    for model_name, robust_acc in lambda_equal.items():
        robust_acc = robust_acc.cpu().numpy()
        for idx, item in enumerate(robust_acc):
            if item >= best_lambda_equal[idx]:
                best_lambda_equal[idx] = item
                best_models_lambda_equal[idx] = model_name

    best_lambda_equal = np.array(best_lambda_equal)
    best_models['lambda_equal'] = best_models_lambda_equal
    # best_lambda_equal = np.max(best_lambda_equal, axis=0)

    # best_lambda_1_zero = [float(0)] * num_epsilons
    # best_models_lambda_1_zero = ['None'] * num_epsilons

    # lambda_1_zero = info['lambda_1_zero']
    # for model_name, robust_acc in lambda_1_zero.items():
    #    robust_acc = robust_acc.cpu().numpy()
    #    for idx, item in enumerate(robust_acc):
    #        if item >= best_lambda_1_zero[idx]:
    #            best_lambda_1_zero[idx] = item
    #            best_models_lambda_1_zero[idx] = model_name

    # best_lambda_1_zero = np.array(best_lambda_1_zero)
    # best_models['lambda_1_zero'] = best_models_lambda_1_zero

    best_lambda_vary = [float(0)] * num_epsilons
    best_models_lambda_vary = ['None'] * num_epsilons

    lambda_vary = info['lambda_vary']
    for model_name, robust_acc in lambda_vary.items():
        # best_lambda_vary.append(robust_acc.cpu().numpy())
        robust_acc = robust_acc.cpu().numpy()
        for idx, item in enumerate(robust_acc):
            if item >= best_lambda_vary[idx]:
                best_lambda_vary[idx] = item
                best_models_lambda_vary[idx] = model_name

    best_lambda_vary = np.array(best_lambda_vary)
    best_models['lambda_vary'] = best_models_lambda_vary

    # plt.plot(epsilons, best_normal, label='normal')
    # plt.plot(epsilons, best_blackout, label='blackout')
    # plt.plot(epsilons, best_lambda_1_zero, label='lambda_1_zero')
    plt.plot(epsilons, best_lambda_equal, label='lambda_equal')
    plt.plot(epsilons, best_lambda_vary, label='lambda_vary')
    plt.legend()
    plt.show()

# save the best_models as a pickle file
with open(best_models_path, 'wb') as write_file:
    pickle.dump(best_models, write_file)

print('Done')
