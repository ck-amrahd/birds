# check the statistical significance of the models on the
# test set
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

test_robust_file1 = 'adversarial/test_robust_exp1.pickle'
test_robust_file2 = 'adversarial/test_robust_exp2.pickle'

robust_1 = pickle.load(open(test_robust_file1, 'rb'))
robust_2 = pickle.load(open(test_robust_file2, 'rb'))

avg_acc = {'lambda_vary' : [],
           'lambda_equal': []}

epsilons = np.linspace(0, 0.2, 10)

for key, val in robust_1.items():
    if key == 'lambda_vary' or key == 'lambda_equal':
        avg_acc[key].append(val)


for key, val in robust_2.items():
    if key == 'lambda_vary' or key == 'lambda_equal':
        v = []
        for idx, item in enumerate(val):
            if idx % 2 == 0:
                v.append(item)
        avg_acc[key].append(v)

lambda_vary = np.array(avg_acc['lambda_vary'])
lambda_equal = np.array(avg_acc['lambda_equal'])
arr = np.vstack((lambda_vary, lambda_equal))
df = pd.DataFrame(arr)
df['categories'] = pd.Series(['vary', 'vary', 'same', 'same'])
print(df)
df.boxplot(by='categories')
plt.show()