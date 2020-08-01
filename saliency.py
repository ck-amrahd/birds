import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import os
import time
import pickle
from pprint import pprint
from utils import bounding_box_grad, load_model
from predict import predict
import multiprocessing
import cv2
from PIL import Image
import torch.nn.functional as f

start = time.time()

# constants
checkpoint_folder = '/home/user/Models/Experiment-4/All/resnet50'
val_folder_path = '/home/user/Models/Experiment-4/data/val'

result_file = 'saliency/saliency.pickle'

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


def calculate_sal(m_name, gp_id, r_dict):
    checkpoint_path = checkpoint_folder + '/' + m_name
    criterion = nn.CrossEntropyLoss()
    model = load_model(checkpoint_path, num_labels, gp_id)
    device = torch.device('cuda:' + gp_id)
    avg_sal = []
    for folder_name in subfolders:
        label = int(folder_name.split('.')[0]) - 1
        target_tensor = torch.tensor([label])
        image_names = os.listdir(val_folder_path + '/' + folder_name)
        for img_name in image_names:
            img_path = val_folder_path + '/' + folder_name + '/' + img_name
            prediction, grad = predict(model, img_path, transform, criterion, target_tensor, height, width,
                                       num_channels, gp_id)
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

    r_dict[m_name] = np.mean(avg_sal)


def generator_from_list(lst, n_size):
    # generates n sized chunk from the given list
    for i in range(0, len(lst), n_size):
        yield lst[i:i + n_size]


# test call for breakpoint
# calculate_sal(all_models[0], '0', dict())
# exit()

manager = multiprocessing.Manager()
return_dict = manager.dict()
jobs = []
for idx, model_name in enumerate(all_models):
    gpu_id = str(idx % 4)
    p = multiprocessing.Process(target=calculate_sal, args=(model_name, gpu_id, return_dict))
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
