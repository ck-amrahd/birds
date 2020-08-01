import pickle
import os
import torch
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from utils import load_model
import torch.nn.functional as f
import torch.nn as nn
from utils import bounding_box_grad
from predict import predict

source_file = 'saliency/saliency.pickle'
test_folder_path = '/home/user/Models/Experiment-4/data/test'

with open(source_file, 'rb') as read_file:
    saliency = pickle.load(read_file)

normal = None
normal_value = None
best_equal = None
best_equal_value = 100.0
best_vary = None
best_vary_value = 100.0

for key, value in saliency.items():
    model_name = key.rsplit('.', 1)[0]
    train_method, lambda_1, lambda_2 = model_name.split('_')

    if train_method == 'normal':
        normal = key
        normal_value = value

    if lambda_1 == lambda_2 and value < best_equal_value:
        best_equal_value = value
        best_equal = key

    if value < best_vary_value:
        best_vary_value = value
        best_vary = key

print('Validation')
print(f'normal: {normal}: {normal_value}')
print(f'best_equal: {best_equal}: {best_equal_value}')
print(f'best_vary: {best_vary}: {best_vary_value}')

height = 224
width = 224
num_channels = 3
num_labels = 200

checkpoint_folder = '/home/user/Models/Experiment-4/All/resnet50'
gpu_id = '0'
device = torch.device('cuda:' + gpu_id if torch.cuda.is_available() else 'cpu')

print('Running on test data...')
transform = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

subfolders = sorted(os.listdir(test_folder_path))


def calculate_saliency(m_name):
    checkpoint_path = checkpoint_folder + '/' + m_name
    model = load_model(checkpoint_path, num_labels, gpu_id)
    model.eval()
    avg_sal = []
    criterion = nn.CrossEntropyLoss()
    for folder_name in subfolders:
        label = int(folder_name.split('.')[0]) - 1
        target_tensor = torch.tensor([label])
        image_names = os.listdir(test_folder_path + '/' + folder_name)
        for img_name in image_names:
            img_path = test_folder_path + '/' + folder_name + '/' + img_name
            prediction, grad = predict(model, img_path, transform, criterion, target_tensor, height, width,
                                       num_channels, gpu_id)
            pred = bounding_box_grad(grad)
            x1, y1, x2, y2 = pred
            original_area = 224 * 224
            crop_area = (x2 - x1) * (y2 - y1)
            a_hat = crop_area / original_area
            a_hat = max(0.05, a_hat)
            orig_img = cv2.imread(img_path)
            orig_img = cv2.resize(orig_img, (height, width))
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            crop_img = orig_img[y1:y2, x1:x2]

            inp = torch.zeros(1, num_channels, height, width)
            img = Image.fromarray(crop_img)
            if img.mode == 'L':
                img = img.convert('RGB')
            img_tensor = transform(img)
            inp[0] = img_tensor

            with torch.no_grad():
                inp = inp.to(device)
                output = model(inp)
                probabilities = f.softmax(output, dim=1)
                prob = probabilities[0, label].item()

            saliency_metric = np.log(a_hat) - np.log(prob)
            avg_sal.append(saliency_metric)

    return np.average(avg_sal)


test_normal = calculate_saliency(normal)
test_equal = calculate_saliency(best_equal)
test_vary = calculate_saliency(best_vary)

print('Test')
print(f'normal: {test_normal}')
print(f'test_equal: {test_equal} ')
print(f'test_vary: {test_vary}')
