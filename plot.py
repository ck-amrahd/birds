import pickle
import matplotlib.pyplot as plt
import os

pickle_folder = 'results/resnet50'
pickle_files  = []
for file_name in os.listdir(pickle_folder):
    if file_name.endswith('.pickle'):
        pickle_files.append(os.path.join(pickle_folder, file_name))


lambda_1 = []
lambda_2 = []
ce_loss = []
penalty_inside = []
penalty_outside = []

for file_name in pickle_files:
    with open(file_name, 'rb') as read_file:
        model_log = pickle.load(read_file)
        lambda_1.append(model_log.get('lambda_1'))
        lambda_2.append(model_log.get('lambda_2'))
        ce_loss.append(model_log.get('train_loss')[0])
        penalty_inside.append(model_log.get('penalty_inside')[0])
        penalty_outside.append(model_log.get('penalty_outside')[0])


plt.scatter(lambda_2, ce_loss, color='blue')
plt.scatter(lambda_2, penalty_outside, color='red')
plt.xlabel('lambda_2')
plt.ylabel('loss and penalty')
plt.show()