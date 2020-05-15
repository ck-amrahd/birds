import pickle
import matplotlib.pyplot as plt

pickle_file_name = '/home/user/Models/Birds/resnet50/bbox_100.0_100.0.pickle'

with open(pickle_file_name, 'rb') as read_file:
    model_log = pickle.load(read_file)
    train_method = model_log.get('train_method')
    lambda_1 = model_log.get('lambda_1')
    lambda_2 = model_log.get('lambda_2')
    ce_loss = model_log.get('train_loss')
    penalty_inside_list = model_log.get('penalty_inside_list')
    penalty_outside_list = model_log.get('penalty_outside_list')
    num_epochs = model_log.get('num_epochs')
    train_acc_list = model_log.get('train_acc_list')
    val_acc_list = model_log.get('val_acc_list')
    train_loss_list = model_log.get('train_loss_list')
    val_loss_list = model_log.get('val_loss_list')


x = list(range(num_epochs))
plt.subplot(221)
plt.plot(x, train_acc_list, label='train_acc_' + train_method)
plt.plot(x, val_acc_list, label='val_acc_' + train_method)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(222)
plt.plot(x, train_loss_list, label='train_loss_' + train_method)
plt.plot(x, val_loss_list, label='val_loss_' + train_method)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(223)
plt.plot(x, penalty_inside_list, label='lambda_1=' + str(lambda_1))
plt.xlabel('Epochs')
plt.ylabel('Penalty-Inside')
plt.legend()

plt.subplot(224)
plt.plot(x, penalty_outside_list, label='lambda_2=' + str(lambda_2))
plt.xlabel('Epochs')
plt.ylabel('Penalty-Outside')
plt.legend()

plt.show()