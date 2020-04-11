import torch
from model import Model
import os
from utils import get_class_id
from torchvision import transforms
from PIL import Image
from bounding_box import BoundingBox
import matplotlib.pyplot as plt
import sys
import pickle
import time
import numpy as np

start = time.time()

if len(sys.argv) != 6:
    print(f'python train.py hpc/local gpu_id bbox/normal/blackout lambda_1, lambda_2')
    sys.exit()

machine = sys.argv[1]
gpu_id = sys.argv[2]
train_method = sys.argv[3]

num_average = 5  # average out the results from 5 runs
num_epochs = 50
learning_rate = 0.01

if machine == 'hpc':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cuda:' + gpu_id if torch.cuda.is_available() else 'cpu')

# lamnda_1 is the regularization we want inside the mask
lambda_1 = float(sys.argv[4])
# lamnda_2 is the regularization we want outside the mask
lambda_2 = float(sys.argv[5])

optimizer = 'SGD'
# optimizer = 'Adam'

model_name = 'resnet50'
start_from_pretrained_model = True
results_folder = 'results'

train_folder_path = 'data/train'
test_folder_path = 'data/test'
images_text_file = 'data/images.txt'
bounding_box_file = 'data/bounding_boxes.txt'
seg_folder = 'data/segmentations'

num_labels = 200
num_channels = 3
height = 224
width = 224

# number of epochs to train for each train_image_indices
batch_size = 32

transform = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

results_folder = results_folder + '/' + model_name
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

checkpoint_path = results_folder + '/' + train_method + '_' + str(lambda_1) + '_' + str(lambda_2) + '.pth'

# model --> model object
# best_model --> best trained model on validataion dataset

# load the train, val and test dataset and pass to different functions
print('Loading required Tensors....\n')
train_images = sorted(os.listdir(train_folder_path))
total_images = len(train_images)

X_train = torch.zeros(total_images, num_channels, height, width)
Y_train = torch.tensor([0] * total_images)

bounding_box = None
if train_method == 'bbox' or train_method == 'blackout':
    # load bounding box information for each training image
    bounding_box = {}
    bbox = BoundingBox(train_folder_path, images_text_file, bounding_box_file, height, width)
    for idx, image_name in enumerate(train_images):
        bounding_box[idx] = bbox.get_bbox(image_name)

# if train with blackout - black out the extra region while loading
if train_method == 'blackout':
    bbox = BoundingBox(train_folder_path, images_text_file, bounding_box_file, height, width)
    for idx, img_name in enumerate(train_images):
        img_path = train_folder_path + '/' + img_name
        img = Image.open(img_path)
        if img.mode == 'L':
            img = img.convert('RGB')

        # img = img.resize((width, height))
        # img = np.array(img)
        # x1, y1, x2, y2 = bounding_box[idx]
        # tmp = np.zeros((height, width, num_channels), dtype=np.uint8)
        # tmp[y1:y2, x1:x2, :] = 1
        # img = img * tmp
        # img = Image.fromarray(img)
        x1, y1, x2, y2 = bbox.get_bbox_unscaled(img_name)
        img = np.array(img)
        img = img[y1:y2, x1:x2, :]
        img = Image.fromarray(img)

        img_tensor = transform(img)

        X_train[idx] = img_tensor
        class_id = get_class_id(img_name)
        Y_train[idx] = int(class_id) - 1  # to make consistent labels to the and test set
else:
    for idx, img_name in enumerate(train_images):
        img_path = train_folder_path + '/' + img_name
        img = Image.open(img_path)
        if img.mode == 'L':
            img = img.convert('RGB')

        img_tensor = transform(img)
        X_train[idx] = img_tensor

        class_id = get_class_id(img_name)
        Y_train[idx] = int(class_id) - 1  # to make consistent labels to the and test set

model = Model(model_name, train_folder_path, test_folder_path, X_train, Y_train, device, num_channels, height,
              width, checkpoint_path, bounding_box, num_labels)

train_image_indices = list(range(len(train_images)))

# run 5 times and average out

acc_list = []
best_acc = 0.0

return_dict = None
for _ in range(num_average):
    return_dict = model.train(train_image_indices, batch_size, num_epochs=num_epochs,
                              train_method=train_method,
                              lambda_1=lambda_1, lambda_2=lambda_2,
                              start_from_pretrained_model=start_from_pretrained_model,
                              learning_rate=learning_rate, optimizer=optimizer,
                              best_acc=best_acc)

    test_acc = return_dict['best_acc_this_run']
    acc_list.append(test_acc)
    if test_acc > best_acc:
        best_acc = test_acc

    print('Best test acc: {:.2f} %'.format(test_acc))

print(f'acc_list: {acc_list}')
avg_best_acc = sum(acc_list) / len(acc_list)
print(f'Average best_acc: {round(avg_best_acc, 2)}')

# dump everything to pickle and save it
train_acc_list = return_dict['train_acc_list']
test_acc_list = return_dict['test_acc_list']
train_loss_list = return_dict['train_loss_list']
test_loss_list = return_dict['test_loss_dict']
penalty_inside_list = return_dict['penalty_inside_list']
penalty_outside_list = return_dict['penalty_outside_list']

model_log = {'num_epochs': num_epochs, 'train_method': train_method,
             'lambda_1': lambda_1, 'lambda_2': lambda_2,
             'train_acc': train_acc_list, 'test_acc': test_acc_list,
             'train_loss': train_loss_list, 'test_loss': test_loss_list,
             'penalty_inside': penalty_inside_list, 'penalty_outside': penalty_outside_list,
             'model_name': model_name, 'start_from_pretrained_model': start_from_pretrained_model,
             'learning_rate': learning_rate, 'optimizer': optimizer,
             'acc_list': acc_list, 'avg_best_acc': avg_best_acc}

log_path = results_folder + '/' + train_method + '_' + str(lambda_1) + '_' + str(lambda_2) + '.pickle'

with open(log_path, 'wb') as write_file:
    pickle.dump(model_log, write_file)

x = list(range(num_epochs))
plt.subplot(221)
plt.plot(x, train_acc_list, label='train_acc_' + train_method)
plt.plot(x, test_acc_list, label='test_acc_' + train_method)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(222)
plt.plot(x, train_loss_list, label='train_loss_' + train_method)
plt.plot(x, test_loss_list, label='test_loss_' + train_method)
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

end = time.time()
elapsed_minutes = (end - start) / 60
print(f'elapsed_minutes: {elapsed_minutes}')
