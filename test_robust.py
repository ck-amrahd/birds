import pickle
import matplotlib.pyplot as plt
import numpy as np

epsilons = np.linspace(0, 0.2, num=10)
test_robust_file_1 = 'adversarial/test_robust_exp1.pickle'
test_robust_file_2 = 'adversarial/test_robust_exp2.pickle'
test_robust_file_3 = 'adversarial/test_robust_exp3.pickle'
test_robust_file_4 = 'adversarial/test_robust_exp4.pickle'
test_robust_file_5 = 'adversarial/test_robust_exp5.pickle'
test_robust_file_6 = 'adversarial/test_robust_exp6.pickle'

test_robust_acc_1 = pickle.load(open(test_robust_file_1, 'rb'))
test_robust_acc_2 = pickle.load(open(test_robust_file_2, 'rb'))
test_robust_acc_3 = pickle.load(open(test_robust_file_3, 'rb'))
test_robust_acc_4 = pickle.load(open(test_robust_file_4, 'rb'))
test_robust_acc_5 = pickle.load(open(test_robust_file_5, 'rb'))
test_robust_acc_6 = pickle.load(open(test_robust_file_6, 'rb'))

for model_class, robust_acc in test_robust_acc_6.items():
    if model_class == 'lambda_vary' or model_class == 'lambda_equal':
        plt.plot(epsilons, robust_acc, label=model_class)

plt.legend()
plt.show()

exit()

# average graph between experiments

robust_acc_vary = []
robust_acc_equal = []

""""
for model_class, robust_acc in test_robust_acc_1.items():
    if model_class == 'lambda_vary':
        robust_acc_vary.append(robust_acc)
    elif model_class == 'lambda_equal':
        robust_acc_equal.append(robust_acc)
    else:
        continue
"""
"""
for model_class, robust_acc in test_robust_acc_2.items():
    if model_class == 'lambda_vary':
        robust_acc = np.array([v for i, v in enumerate(robust_acc) if i % 2 == 0])
        robust_acc_vary.append(robust_acc)
    elif model_class == 'lambda_equal':
        robust_acc = np.array([v for i, v in enumerate(robust_acc) if i % 2 == 0])
        robust_acc_equal.append(robust_acc)
    else:
        continue

for model_class, robust_acc in test_robust_acc_3.items():
    if model_class == 'lambda_vary':
        robust_acc = np.array([v for i, v in enumerate(robust_acc) if i % 2 == 0])
        robust_acc_vary.append(robust_acc)
    elif model_class == 'lambda_equal':
        robust_acc = np.array([v for i, v in enumerate(robust_acc) if i % 2 == 0])
        robust_acc_equal.append(robust_acc)
    else:
        continue

for model_class, robust_acc in test_robust_acc_4.items():
    if model_class == 'lambda_vary':
        robust_acc = np.array([v for i, v in enumerate(robust_acc) if i % 2 == 0])
        robust_acc_vary.append(robust_acc)
    elif model_class == 'lambda_equal':
        robust_acc = np.array([v for i, v in enumerate(robust_acc) if i % 2 == 0])
        robust_acc_equal.append(robust_acc)
    else:
        continue
"""

for model_class, robust_acc in test_robust_acc_5.items():
    if model_class == 'lambda_vary':
        robust_acc_vary.append(robust_acc)
    elif model_class == 'lambda_equal':
        robust_acc_equal.append(robust_acc)
    else:
        continue

for model_class, robust_acc in test_robust_acc_6.items():
    if model_class == 'lambda_vary':
        robust_acc_vary.append(robust_acc)
    elif model_class == 'lambda_equal':
        robust_acc_equal.append(robust_acc)
    else:
        continue

robust_acc_vary = np.vstack(robust_acc_vary)
robust_acc_equal = np.vstack(robust_acc_equal)

# robust_acc_vary = np.mean(robust_acc_vary, axis=0)
# robust_acc_equal = np.mean(robust_acc_equal, axis=0)
# plt.plot(epsilons, robust_acc_equal, label='lambda_equal')
# plt.plot(epsilons, robust_acc_vary, label='lambda_vary')
# plt.legend()
# plt.show()

vary_mean = np.mean(robust_acc_vary, axis=0)
vary_std = np.std(robust_acc_vary, axis=0)

equal_mean = np.mean(robust_acc_equal, axis=0)
equal_std = np.std(robust_acc_equal, axis=0)

fig, ax = plt.subplots()
ax.plot(epsilons, vary_mean)
ax.fill_between(epsilons, (vary_mean - vary_std), (vary_mean + vary_std), color='b', alpha=0.2)

ax.plot(epsilons, equal_mean)
ax.fill_between(epsilons, (equal_mean - equal_std), (equal_mean + equal_std), color='r', alpha=0.2)

plt.show()
