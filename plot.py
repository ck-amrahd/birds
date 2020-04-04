import pickle
import matplotlib.pyplot as plt
import os
import math

pickle_folder = 'results/resnet50-lambda1-0'
pickle_files  = []
for file_name in os.listdir(pickle_folder):
    if file_name.endswith('.pickle'):
        pickle_files.append(os.path.join(pickle_folder, file_name))


lambda_1 = []
lambda_2 = []
ce_loss = []
penalty_inside = []
penalty_outside = []
best_ac_list = []

for file_name in pickle_files:
    with open(file_name, 'rb') as read_file:
        model_log = pickle.load(read_file)
        lambda_1.append(model_log.get('lambda_1'))
        lambda_2.append(model_log.get('lambda_2'))
        ce_loss.append(model_log.get('train_loss')[-1])
        # penalty_inside.append(model_log.get('penalty_inside')[-1] / model_log.get('lambda_1'))
        penalty_outside.append(model_log.get('penalty_outside')[-1] / model_log.get('lambda_2'))

        test_acc = model_log.get('test_acc')
        best_acc = round(max(test_acc), 2)
        best_ac_list.append(best_acc)


lambda_2 = [math.log(x) for x in lambda_2]
plt.scatter(lambda_2, ce_loss, color='blue', label='ce_loss')
# plt.scatter(lambda_1, penalty_inside, color='red', label='penalty')
plt.scatter(lambda_2, penalty_outside, color='red', label='penalty')
# plt.scatter(lambda_2, best_ac_list, label='best_acc')
# plt.xlabel('lambda_1')
plt.xlabel('log(lambda_2)')
plt.ylabel('loss and penalty')
plt.legend()
plt.show()