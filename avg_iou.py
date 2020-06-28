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
checkpoint_folder = '/home/user/Models/Experiment-4/All/resnet50'
val_folder_path = '/home/user/Models/Experiment-4/data/val'

result_file = 'iou/avg_iou.pickle'

images_text_file = 'data/images.txt'
bounding_box_file = 'data/bounding_boxes.txt'

height = 224
width = 224
num_channels = 3
num_labels = 200

all_models = []
for file_name in os.listdir(checkpoint_folder):
    if file_name.endswith('.pth'):
        all_models.append(file_name)

print('Running...')
print(f'Total models: {len(all_models)}')
transform = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

subfolders = sorted(os.listdir(val_folder_path))
bbox = BoundingBox(val_folder_path, images_text_file, bounding_box_file, height, width)


def calculate_iou(m_name, gp_id, r_dict):
    checkpoint_path = checkpoint_folder + '/' + m_name
    criterion = nn.CrossEntropyLoss()
    model = load_model(checkpoint_path, num_labels, gp_id)
    average_iou = []
    for folder_name in subfolders:
        label = int(folder_name.split('.')[0]) - 1
        target_tensor = torch.tensor([label])
        image_names = os.listdir(val_folder_path + '/' + folder_name)
        for img_name in image_names:
            img_path = val_folder_path + '/' + folder_name + '/' + img_name
            x1_gt, y1_gt, x2_gt, y2_gt = bbox.get_bbox_from_path(img_path)
            gt = [x1_gt, y1_gt, x2_gt, y2_gt]
            grad = predict(model, img_path, transform, criterion, target_tensor, height, width,
                           num_channels, gp_id)
            pred = bounding_box_grad(grad)
            overlap = check_overlap(pred, gt)
            if overlap:
                iou = get_iou(pred, gt)
            else:
                iou = 0.0

            average_iou.append(iou)

    # print(f'{m_name}: {np.mean(average_iou)}')
    r_dict[m_name] = np.mean(average_iou)


def generator_from_list(lst, n):
    # generates n sized chunk from the given list
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


manager = multiprocessing.Manager()
return_dict = manager.dict()
jobs = []
for idx, model_name in enumerate(all_models):
    gpu_id = str(idx % 4)
    p = multiprocessing.Process(target=calculate_iou, args=(model_name, gpu_id, return_dict))
    jobs.append(p)

n = 4
counter = 0
for chunks in generator_from_list(jobs, n):
    for proc in chunks:
        proc.start()

    for p in chunks:
        p.join()

    counter += n
    print(f'{counter} models done.')

with open(result_file, 'wb') as write_file:
    pickle.dump(dict(return_dict), write_file)

pprint(dict(return_dict))

end = time.time()
elapsed_minutes = (end - start) / 60
print(f'elapsed-minutes: {round(elapsed_minutes, 2)}')
