import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import cv2
from bounding_box import BoundingBox
import torch.nn.functional as F
import os
from utils import load_model
import pickle
from pprint import pprint
import time


start = time.time()
val_folder_path = '/home/user/Models/Experiment-4/data/val'
images_text_file = 'data/images.txt'
bounding_box_file = 'data/bounding_boxes.txt'
height = 224
width = 224
num_labels = 200
result_file = 'saliency/saliency.pickle'
checkpoint_folder = '/home/user/Models/Experiment-4/All/resnet50'
gpu_id = '0'
device = torch.device('cuda:' + gpu_id if torch.cuda.is_available() else 'cpu')

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

bbox = BoundingBox(val_folder_path, images_text_file, bounding_box_file, height, width)
subfolders = sorted(os.listdir(val_folder_path))


def preprocess(im_path):
    x1_gt, y1_gt, x2_gt, y2_gt = bbox.get_bbox_from_path_unscaled(im_path)
    orig_img = cv2.imread(im_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    orig_height, orig_width, orig_channels = orig_img.shape
    original_area = orig_height * orig_width
    crop_img = orig_img[y1_gt:y2_gt, x1_gt:x2_gt]
    crop_img = cv2.resize(crop_img, (height, width))
    crop_area = height * width
    # a_hat is the ratio of cropped region to total area
    a_hat = crop_area / original_area

    inp = torch.zeros(1, 3, 224, 224)
    # img = Image.open(img_path)
    img = Image.fromarray(crop_img)
    if img.mode == 'L':
        img = img.convert('RGB')
    img_tensor = transform(img)
    inp[0] = img_tensor
    return inp, a_hat


results = {}
for model_name in all_models:
    checkpoint_path = checkpoint_folder + '/' + model_name
    model = load_model(checkpoint_path, num_labels, gpu_id)
    model.eval()
    avg_sal = []
    inputs, a_hat = None, None
    for folder_name in subfolders:
        label = int(folder_name.split('.')[0]) - 1
        target_tensor = torch.tensor([label])
        image_names = os.listdir(val_folder_path + '/' + folder_name)
        for img_name in image_names:
            img_path = val_folder_path + '/' + folder_name + '/' + img_name
            inputs, a_hat = preprocess(img_path)
        with torch.no_grad():
            inputs = inputs.to(device)
            output = model(inputs)
            probabilities = F.softmax(output, dim=1)
            p = probabilities[0, label].item()

        saliency_metric = np.log(a_hat) - np.log(p)
        avg_sal.append(saliency_metric)

    results[model_name] = np.average(avg_sal)

with open(result_file, 'wb') as write_file:
    pickle.dump(results, write_file)

pprint(results)
end = time.time()
minutes = (end - start) / 60.0
print(f'Time: {round(minutes, 2)} Minutes')