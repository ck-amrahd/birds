import pickle
import matplotlib.pyplot as plt
import numpy as np

epsilons = np.linspace(0, 0.2, num=20)
test_robust_file = 'adversarial/test_robust_exp2.pickle'

test_robust_acc = pickle.load(open(test_robust_file, 'rb'))

for model_class, robust_acc in test_robust_acc.items():
    if model_class == 'lambda_vary' or model_class == 'lambda_equal':
        plt.plot(epsilons, robust_acc, label=model_class)

plt.legend()
plt.show()