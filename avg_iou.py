import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from bounding_box import BoundingBox
import os
import time
import pickle
from pprint import pprint
from utils import check_overlap, get_iou, bounding_box_grad, load_model
from predict import predict
import multiprocessing

start = time.time()

# constants
results_file = 'avg_iou.pickle'
test_folder_path = 'data/test'
images_text_file = 'data/images.txt'
bounding_box_file = 'data/bounding_boxes.txt'
height = 224
width = 224
num_channels = 3
num_labels = 200
gpu_id = 3

checkpoint_folder = '/home/user/Models/Experiment-4/All/resnet50'
all_models = []
for file_name in os.listdir(checkpoint_folder):
    if file_name.endswith('.pth'):
        all_models.append(file_name)

print('Running...')
transform = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

subfolders = sorted(os.listdir(test_folder_path))
bbox = BoundingBox(test_folder_path, images_text_file, bounding_box_file, height, width)


def calculate_iou(m_name, m_iou):
    checkpoint_path = checkpoint_folder + '/' + m_name
    criterion = nn.CrossEntropyLoss()
    model = load_model(checkpoint_path, num_labels)
    average_iou = []
    for folder_name in subfolders:
        label = int(folder_name.split('.')[0]) - 1
        target_tensor = torch.tensor([label])
        image_names = os.listdir(test_folder_path + '/' + folder_name)
        for img_name in image_names:
            img_path = test_folder_path + '/' + folder_name + '/' + img_name
            x1_gt, y1_gt, x2_gt, y2_gt = bbox.get_bbox_from_path(img_path)
            gt = [x1_gt, y1_gt, x2_gt, y2_gt]
            grad = predict(model, img_path, transform, criterion, target_tensor, height, width, num_channels)
            pred = bounding_box_grad(grad)
            overlap = check_overlap(pred, gt)
            if overlap:
                iou = get_iou(pred, gt)
            else:
                iou = 0.0

            average_iou.append(iou)

    m_iou[model_name] = np.mean(average_iou)


manager = multiprocessing.Manager()
models_iou = manager.dict()
jobs = []
for model_name in all_models:
    # avg_iou = calculate_iou(model_name)
    # models_iou[model_name] = avg_iou
    p = multiprocessing.Process(target=calculate_iou, args=(model_name, models_iou))
    jobs.append(p)
    p.start()

for proc in jobs:
    proc.join()

with open(results_file, 'wb') as write_file:
    pickle.dump(models_iou, write_file)

pprint(models_iou)

end = time.time()
elapsed_minutes = (end - start) / 60
print(f'elapsed-minutes: {round(elapsed_minutes, 2)}')
