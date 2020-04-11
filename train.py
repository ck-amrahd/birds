import torch
from model import Model
import os
from utils import get_class_id
from torchvision import transforms
from PIL import Image
from bounding_box import BoundingBox
import matplotlib.pyplot as plt
from segmentations import Segmentation
import sys
import pickle
import time


start = time.time()

if len(sys.argv) != 5:
    print(f'python train.py gpu_id seg/bbox/normal lambda_1, lambda_2')
    sys.exit()

gpu_id = sys.argv[1]
device = torch.device('cuda:' + gpu_id if torch.cuda.is_available() else 'cpu')
# lamnda_1 is the regularization we want inside the mask
lambda_1 = float(sys.argv[3])
# lamnda_2 is the regularization we want outside the mask
lambda_2 = float(sys.argv[4])

num_epochs = 50
learning_rate = 0.01
optimizer = 'SGD'
# optimizer = 'Adam'

train_with_bounding_box = False
train_with_seg_mask = False

if sys.argv[2] == 'bbox':
    train_with_bounding_box = True

if sys.argv[2] == 'seg':
    train_with_seg_mask = True

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

if train_with_bounding_box:
    checkpoint_path = results_folder + '/' + 'model_bbox_' + str(lambda_1) + '_' + str(lambda_2) + '.pth'
elif train_with_seg_mask:
    checkpoint_path = results_folder + '/' + 'model_seg_' + str(lambda_1) + '_' + str(lambda_2) + '.pth'
else:
    checkpoint_path = results_folder + '/' + 'model_normal_' + str(lambda_1) + '_' + str(lambda_2) + '.pth'

# model --> model object
# best_model --> best trained model on validataion dataset

# load the train, val and test dataset and pass to different functions
print('Loading required Tensors....\n')
train_images = sorted(os.listdir(train_folder_path))
total_images = len(train_images)

X_train = torch.zeros(total_images, num_channels, height, width)
Y_train = torch.tensor([0] * total_images)

for idx, img_name in enumerate(train_images):
    img_path = train_folder_path + '/' + img_name
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')

    img_tensor = transform(img)
    X_train[idx] = img_tensor

    class_id = get_class_id(img_name)
    Y_train[idx] = int(class_id) - 1  # to make consistent labels to the and test set

if train_with_bounding_box:
    # load bounding box information for each training image
    bouding_box = {}
    bbox = BoundingBox(train_folder_path, images_text_file, bounding_box_file, height, width)
    for idx, image_name in enumerate(train_images):
        bouding_box[idx] = bbox.get_bbox(image_name)
else:
    bouding_box = None

if train_with_seg_mask:
    # bird location will have value 255.0 and the other locations will have value 0.0
    segmentation_mask = {}
    seg = Segmentation(train_folder_path, images_text_file, seg_folder, height, width)
    for idx, img_name in enumerate(train_images):
        segmentation_mask[idx] = seg.get_segmentation_mask(img_name)
else:
    segmentation_mask = None

model = Model(model_name, train_folder_path, test_folder_path, X_train, Y_train, device, num_channels, height,
              width, checkpoint_path, bouding_box, segmentation_mask, num_labels)

train_image_indices = list(range(len(train_images)))

# run 5 times and average out

acc_list = []
best_acc = 0.0

for i in range(5):
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []
    test_acc, p_inside, p_outside = model.train(train_image_indices, batch_size, train_acc_list,
                                                test_acc_list, train_loss_list,
                                                test_loss_list, num_epochs=num_epochs,
                                                train_with_bbox=train_with_bounding_box,
                                                train_with_seg_mask=train_with_seg_mask,
                                                lambda_1=lambda_1, lambda_2=lambda_2,
                                                start_from_pretrained_model=start_from_pretrained_model,
                                                learning_rate=learning_rate, optimizer=optimizer,
                                                best_acc=best_acc)

    acc_list.append(test_acc)
    if test_acc > best_acc:
        best_acc = test_acc

    print('Best test acc: {:.2f} %'.format(test_acc))

print(f'acc_list: {acc_list}')
avg_best_acc = sum(acc_list) / len(acc_list)
print(f'Average best_acc: {round(avg_best_acc, 2)}')

# dump everything to pickle and save it
model_log = {'num_epochs': num_epochs, 'train_method': sys.argv[2],
             'lambda_1': lambda_1, 'lambda_2': lambda_2,
             'train_acc': train_acc_list, 'test_acc': test_acc_list,
             'train_loss': train_loss_list, 'test_loss': test_loss_list,
             'penalty_inside': p_inside, 'penalty_outside': p_outside,
             'model_name': model_name, 'start_from_pretrained_model': start_from_pretrained_model,
             'learning_rate': learning_rate, 'optimizer': optimizer,
             'acc_list': acc_list, 'avg_best_acc': avg_best_acc}

if train_with_bounding_box:
    log_path = results_folder + '/' + 'model_bbox_' + str(lambda_1) + '_' + str(lambda_2) + '.pickle'
elif train_with_seg_mask:
    log_path = results_folder + '/' + 'model_seg_' + str(lambda_1) + '_' + str(lambda_2) + '.pickle'
else:
    log_path = results_folder + '/' + 'model_normal_' + str(lambda_1) + '_' + str(lambda_2) + '.pickle'

with open(log_path, 'wb') as write_file:
    pickle.dump(model_log, write_file)

x = list(range(num_epochs))
plt.subplot(221)
plt.plot(x, train_acc_list, label='train_acc_' + sys.argv[2])
plt.plot(x, test_acc_list, label='test_acc_' + sys.argv[2])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(222)
plt.plot(x, train_loss_list, label='train_loss_' + sys.argv[2])
plt.plot(x, test_loss_list, label='test_loss_' + sys.argv[2])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(223)
plt.plot(x, p_inside, label='lambda_1=' + str(lambda_1))
plt.xlabel('Epochs')
plt.ylabel('Penalty-Inside')
plt.legend()

plt.subplot(224)
plt.plot(x, p_outside, label='lambda_2=' + str(lambda_2))
plt.xlabel('Epochs')
plt.ylabel('Penalty-Outside')
plt.legend()

plt.show()

end = time.time()
elapsed_minutes = (end - start) / 60
print(f'elapsed_minutes: {elapsed_minutes}')
