import pickle
import matplotlib.pyplot as plt
import numpy as np

pickle_file_name = 'results/resnet50/model_bbox_0.0_100.0.pickle'

with open(pickle_file_name, 'rb') as read_file:
    model_log = pickle.load(read_file)
    train_method = model_log.get('train_method')
    lambda_1 = model_log.get('lambda_1')
    lambda_2 = model_log.get('lambda_2')
    ce_loss = model_log.get('train_loss')
    penalty_inside = model_log.get('penalty_inside')
    penalty_outside = model_log.get('penalty_outside')
    num_epochs = model_log.get('num_epochs')
    train_acc = model_log.get('train_acc')
    test_acc = model_log.get('test_acc')
    train_loss = model_log.get('train_loss')
    test_loss = model_log.get('test_loss')

x = list(range(num_epochs))
f = plt.figure(figsize=(12, 8))
ax1 = f.add_subplot(121)
ax1.plot(x, train_loss, label='train_loss')
ax1.plot(x, test_loss, label='test_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
ax1.legend()

# normalize penalty_outside by lambda_2
penalty_outside = np.array(penalty_outside) / lambda_2

ax2 = f.add_subplot(122)
ax2.plot(x, penalty_outside, label='lambda_2=' + str(lambda_2))
plt.xlabel('Epochs')
plt.ylabel('Penalty-Outside')
ax2.legend()

plt.savefig(train_method + '_' + str(lambda_1) + '_' + str(lambda_2) + '.png', dpi=800)
plt.show()