# check the statistical significance of the models on the
# test set
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

test_robust_file1 = 'adversarial/test_robust_exp1.pickle'
test_robust_file2 = 'adversarial/test_robust_exp2.pickle'
test_robust_file3 = 'adversarial/test_robust_exp3.pickle'

robust_1 = pickle.load(open(test_robust_file1, 'rb'))
robust_2 = pickle.load(open(test_robust_file2, 'rb'))
robust_3 = pickle.load(open(test_robust_file3, 'rb'))

avg_acc = {'lambda_vary': [],
           'lambda_equal': []}

epsilons = np.linspace(0, 0.2, 10)

for key, val in robust_1.items():
    if key == 'lambda_vary' or key == 'lambda_equal':
        avg_acc[key].append(val)

for key, val in robust_2.items():
    if key == 'lambda_vary' or key == 'lambda_equal':
        half_array = np.array([v for i, v in enumerate(val) if i % 2 == 0])
        avg_acc[key].append(half_array)

for key, val in robust_2.items():
    if key == 'lambda_vary' or key == 'lambda_equal':
        half_array = np.array([v for i, v in enumerate(val) if i % 2 == 0])
        avg_acc[key].append(half_array)

lambda_vary = np.array(avg_acc['lambda_vary'])
lambda_equal = np.array(avg_acc['lambda_equal'])
arr = np.vstack((lambda_vary, lambda_equal))
column_names = [str(round(e, 2)) for e in epsilons]
df = pd.DataFrame(arr, columns=column_names)
df['categories'] = pd.Series(['vary', 'vary', 'vary', 'same', 'same', 'same'])

print(df)

# df.boxplot(column=[str(round(e, 2)) for e in epsilons], by='categories')
# plt.show()

# for i in range(0, len(epsilons)):
#    df.boxplot(column=column_names[i], by='categories', fontsize=1.5)
#    plt.show()

df_long = pd.melt(df, 'categories', var_name='epsilons', value_name='robust_acc')
sns.boxplot(x='epsilons', hue='categories', y='robust_acc', data=df_long)
plt.show()
