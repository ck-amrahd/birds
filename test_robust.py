import pickle
import matplotlib.pyplot as plt
import numpy as np

epsilons = np.linspace(0, 0.2, num=10)

test_robust_file_2 = '/home/user/Models/birds/Experiment-2/Results/test_robust_exp2.pickle'
test_robust_file_3 = '/home/user/Models/birds/Experiment-3/Results/test_robust_exp3.pickle'
test_robust_file_4 = '/home/user/Models/birds/Experiment-4/Results/test_robust_exp4.pickle'

test_robust_acc_2 = pickle.load(open(test_robust_file_2, 'rb'))
test_robust_acc_3 = pickle.load(open(test_robust_file_3, 'rb'))
test_robust_acc_4 = pickle.load(open(test_robust_file_4, 'rb'))

"""
for model_class, robust_acc in test_robust_acc_4.items():
    # if model_class == 'lambda_vary' or model_class == 'lambda_equal':
    robust_acc = np.array([v for i, v in enumerate(robust_acc) if i % 2 == 0])
    plt.plot(epsilons, robust_acc, label=model_class)

plt.legend()
plt.show()

exit()
"""

robust_acc_vary = []
robust_acc_equal = []
robust_acc_normal = []
robust_acc_blackout = []

for model_class, robust_acc in test_robust_acc_2.items():
    if model_class == 'lambda_vary':
        robust_acc = np.array([v for i, v in enumerate(robust_acc) if i % 2 == 0])
        robust_acc_vary.append(robust_acc)
    elif model_class == 'lambda_equal':
        robust_acc = np.array([v for i, v in enumerate(robust_acc) if i % 2 == 0])
        robust_acc_equal.append(robust_acc)
    elif model_class == 'normal':
        robust_acc = np.array([v for i, v in enumerate(robust_acc) if i % 2 == 0])
        robust_acc_normal.append(robust_acc)
    elif model_class == 'blackout':
        robust_acc = np.array([v for i, v in enumerate(robust_acc) if i % 2 == 0])
        robust_acc_blackout.append(robust_acc)
    else:
        continue

for model_class, robust_acc in test_robust_acc_3.items():
    if model_class == 'lambda_vary':
        robust_acc = np.array([v for i, v in enumerate(robust_acc) if i % 2 == 0])
        robust_acc_vary.append(robust_acc)
    elif model_class == 'lambda_equal':
        robust_acc = np.array([v for i, v in enumerate(robust_acc) if i % 2 == 0])
        robust_acc_equal.append(robust_acc)
    elif model_class == 'normal':
        robust_acc = np.array([v for i, v in enumerate(robust_acc) if i % 2 == 0])
        robust_acc_normal.append(robust_acc)
    elif model_class == 'blackout':
        robust_acc = np.array([v for i, v in enumerate(robust_acc) if i % 2 == 0])
        robust_acc_blackout.append(robust_acc)
    else:
        continue

for model_class, robust_acc in test_robust_acc_4.items():
    if model_class == 'lambda_vary':
        robust_acc = np.array([v for i, v in enumerate(robust_acc) if i % 2 == 0])
        robust_acc_vary.append(robust_acc)
    elif model_class == 'lambda_equal':
        robust_acc = np.array([v for i, v in enumerate(robust_acc) if i % 2 == 0])
        robust_acc_equal.append(robust_acc)
    elif model_class == 'normal':
        robust_acc = np.array([v for i, v in enumerate(robust_acc) if i % 2 == 0])
        robust_acc_normal.append(robust_acc)
    elif model_class == 'blackout':
        robust_acc = np.array([v for i, v in enumerate(robust_acc) if i % 2 == 0])
        robust_acc_blackout.append(robust_acc)
    else:
        continue

robust_acc_vary = np.vstack(robust_acc_vary)
robust_acc_equal = np.vstack(robust_acc_equal)
robust_acc_normal = np.vstack(robust_acc_normal)
robust_acc_blackout = np.vstack(robust_acc_blackout)

vary_mean = np.mean(robust_acc_vary, axis=0)
vary_std = np.std(robust_acc_vary, axis=0)

equal_mean = np.mean(robust_acc_equal, axis=0)
equal_std = np.std(robust_acc_equal, axis=0)

normal_mean = np.mean(robust_acc_normal, axis=0)
normal_std = np.std(robust_acc_normal, axis=0)

blackout_mean = np.mean(robust_acc_blackout, axis=0)
blackout_std = np.std(robust_acc_blackout, axis=0)

fig, ax = plt.subplots()
ax.plot(epsilons, vary_mean, label='lambda vary')
ax.fill_between(epsilons, (vary_mean - vary_std), (vary_mean + vary_std), alpha=0.3)

ax.plot(epsilons, equal_mean, label='lambda equal')
ax.fill_between(epsilons, (equal_mean - equal_std), (equal_mean + equal_std), alpha=0.3)

ax.plot(epsilons, normal_mean, label='normal')
ax.fill_between(epsilons, (normal_mean - normal_std), (normal_mean + normal_std), alpha=0.3)

ax.plot(epsilons, blackout_mean, label='blackout')
ax.fill_between(epsilons, (blackout_mean - blackout_std), (blackout_mean + blackout_std), alpha=0.3)

plt.legend()
plt.show()
