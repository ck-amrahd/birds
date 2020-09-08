# load all_info pickle file and generate graphs as necessary
import pickle
import numpy as np
import matplotlib.pyplot as plt

info_file = 'adversarial/all_info_exp2.pickle'
num_epsilons = 10
epsilons = np.linspace(0, 1, num=num_epsilons)

# results file - that stores best model for each value of epsilon
best_models_path = 'adversarial/best_models_exp2.pickle'
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

    lambda_vary = info['lambda_vary']
    best_lambda_vary = [float(0)] * num_epsilons
    best_models_lambda_vary = ['None'] * num_epsilons
    for model_name, robust_acc in lambda_vary.items():
        robust_acc = robust_acc.cpu().numpy()
        for idx, item in enumerate(robust_acc):
            if item >= best_lambda_vary[idx]:
                best_lambda_vary[idx] = item
                best_models_lambda_vary[idx] = model_name

    lambda1_zero = info['lambda1_zero']
    best_lambda1_zero = [float(0)] * num_epsilons
    best_models_lambda1_zero = ['None'] * num_epsilons
    for model_name, robust_acc in lambda1_zero.items():
        robust_acc = robust_acc.cpu().numpy()
        for idx, item in enumerate(robust_acc):
            if item >= best_lambda1_zero[idx]:
                best_lambda1_zero[idx] = item
                best_models_lambda1_zero[idx] = model_name

            # because lambda_vary includes lambda1_zero
            if item >= best_lambda_vary[idx]:
                best_lambda_vary[idx] = item
                best_models_lambda_vary[idx] = model_name

    best_lambda1_zero = np.array(best_lambda1_zero)
    best_models['lambda1_zero'] = best_models_lambda1_zero

    best_lambda_equal = [float(0)] * num_epsilons
    best_models_lambda_equal = ['None'] * num_epsilons

    lambda_equal = info['lambda_equal']
    for model_name, robust_acc in lambda_equal.items():
        robust_acc = robust_acc.cpu().numpy()
        for idx, item in enumerate(robust_acc):
            if item >= best_lambda_equal[idx]:
                best_lambda_equal[idx] = item
                best_models_lambda_equal[idx] = model_name

            # because lambda_vary includes lambda_equal
            if item >= best_lambda_vary[idx]:
                best_lambda_vary[idx] = item
                best_models_lambda_vary[idx] = model_name

    best_lambda_equal = np.array(best_lambda_equal)
    best_models['lambda_equal'] = best_models_lambda_equal

    # convert for best_vary here
    best_lambda_vary = np.array(best_lambda_vary)
    best_models['lambda_vary'] = best_models_lambda_vary

    plt.plot(epsilons, best_normal, label='normal')
    plt.plot(epsilons, best_blackout, label='blackout')
    plt.plot(epsilons, best_lambda1_zero, label='lambda1_zero')
    plt.plot(epsilons, best_lambda_equal, label='lambda_equal')
    plt.plot(epsilons, best_lambda_vary, label='lambda_vary')
    plt.legend()
    plt.show()

# save the best_models as a pickle file
with open(best_models_path, 'wb') as write_file:
    pickle.dump(best_models, write_file)

print('Done')

