import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pickle_path = 'iou/avg_iou_loc.pickle'

with open(pickle_path, 'rb') as read_file:
    avg_iou_loc = pickle.load(read_file)

value_index = {'0.0': 0, '0.1': 1, '0.22': 2, '0.46': 3, '1.0': 4, '2.15': 5, '4.64': 6, '10.0': 7, '21.54': 8,
               '46.42': 9, '100.0': 10, '215.44': 11, '464.16': 12, '1000.0': 13}

heatmap_array = np.zeros((14, 14))
for model_name, iou_loc in avg_iou_loc.items():
    model_name = model_name.rsplit('.', 1)[0]
    train_method, lambda_1, lambda_2 = model_name.split('_')
    heatmap_array[value_index[lambda_1]][value_index[lambda_2]] = round(iou_loc[1], 2)

plt.figure(figsize=(19.20, 10.80))
x_axis_labels = [0.0, 0.1, 0.22, 0.46, 1.0, 2.15, 4.64, 10.0, 21.54, 46.42, 100.0, 215.44, 464.16, 1000.0]
y_axis_labels = [0.0, 0.1, 0.22, 0.46, 1.0, 2.15, 4.64, 10.0, 21.54, 46.42, 100.0, 215.44, 464.16, 1000.0]
ax = sns.heatmap(heatmap_array, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels, linewidths=0.5)
ax.set(xlabel='lambda_2', ylabel='lambda_1')
plt.savefig('iou/loc_exp4.png', format='png')
plt.show()
