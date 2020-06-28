import pickle
import torch
from torchvision import transforms
import os
from bounding_box import BoundingBox
from utils import check_overlap, get_iou, load_model
from torch import nn
from predict import predict
from utils import bounding_box_grad
from pprint import pprint

pickle_path = 'iou/avg_iou_loc.pickle'

with open(pickle_path, 'rb') as read_file:
    avg_iou_loc = pickle.load(read_file)

# constants
checkpoint_folder = '/home/user/Models/Experiment-4/All/resnet50'
test_folder_path = '/home/user/Models/Experiment-4/data/test'

result_file = 'iou/test_loc.pickle'

images_text_file = 'data/images.txt'
bounding_box_file = 'data/bounding_boxes.txt'

height = 224
width = 224
num_channels = 3
num_labels = 200

print('Running...')
transform = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

subfolders = sorted(os.listdir(test_folder_path))
bbox = BoundingBox(test_folder_path, images_text_file, bounding_box_file, height, width)

best_equal = None
best_vary = None
best_equal_value = -1.0
best_vary_value = -1.0
normal_method = None
normal_value = None

for key, value in avg_iou_loc.items():
    model_name = key.rsplit('.', 1)[0]
    train_method, lambda_1, lambda_2 = model_name.split('_')

    value = value[1]
    if lambda_1 == lambda_2 and value > best_equal_value:
        best_equal_value = value
        best_equal = key

    if value > best_vary_value:
        best_vary_value = value
        best_vary = key

    if train_method == 'normal':
        normal_value = value
        normal_method = key

print(f'normal: {normal_method} : {normal_value}')
print(f'best_equal: {best_equal} : {best_equal_value}')
print(f'best_vary: {best_vary} : {best_vary_value}')
print('Running on test data...')


def calculate_loc(m_name, gp_id):
    checkpoint_path = checkpoint_folder + '/' + m_name
    criterion = nn.CrossEntropyLoss()
    model = load_model(checkpoint_path, num_labels, gp_id)
    correct = 0
    total = 0
    for folder_name in subfolders:
        label = int(folder_name.split('.')[0]) - 1
        target_tensor = torch.tensor([label])
        image_names = os.listdir(test_folder_path + '/' + folder_name)
        for img_name in image_names:
            img_path = test_folder_path + '/' + folder_name + '/' + img_name
            x1_gt, y1_gt, x2_gt, y2_gt = bbox.get_bbox_from_path(img_path)
            gt = [x1_gt, y1_gt, x2_gt, y2_gt]
            prediction, grad = predict(model, img_path, transform, criterion, target_tensor, height, width,
                                       num_channels, gp_id)
            pred = bounding_box_grad(grad)
            overlap = check_overlap(pred, gt)
            if overlap:
                iou = get_iou(pred, gt)
            else:
                iou = 0.0

            total += 1
            if prediction == label and iou >= 0.5:
                correct += 1

    acc = float(correct) / float(total)
    return acc


test_normal = calculate_loc(normal_method, '1')
test_equal = calculate_loc(best_equal, '0')
test_vary = calculate_loc(best_vary, '3')

result = {'test_equal': test_equal, 'test_vary': test_vary, 'test_normal': test_normal}
pprint(result)

with open(result_file, 'wb') as write_file:
    pickle.dump(result, write_file)
