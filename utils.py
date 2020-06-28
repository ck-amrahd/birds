import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from torchvision import models
import torch
from torch import nn

train_folder_path = 'data/train'
class_file_path = 'data/classes.txt'
img_file = 'data/images.txt'

classNameToId = {}
with open(class_file_path, 'r') as read_file:
    for line in read_file:
        line = line.strip()
        class_id, class_name = line.split(' ')
        # put the name only in class_name instead of 001.Black_footed_Albatross, just put
        # black_footed_albatross --> all lowercase
        class_name = class_name.split('.')[1].lower()
        classNameToId[class_name] = class_id.strip()

imageToIndex = {}
indexToImage = {}

# os.listdir returns in arbitrary order so, sort them such that this index will be same for
# indexHandler and here
training_images = sorted(os.listdir(train_folder_path))
for i, file_name in enumerate(training_images):
    indexToImage[i] = file_name
    imageToIndex[file_name] = i


def get_image_names_from_index(list_indices):
    return list(map(lambda x: indexToImage[x], list_indices))


def get_image_name_from_index(idx):
    return indexToImage[idx]


img_name_to_id = {}

with open(img_file, 'r') as read_file:
    for line in read_file:
        img_id, img_name = line.strip().split(' ')
        img_name = img_name.split('/')[-1]
        img_name_to_id[img_name] = img_id


def get_image_id_from_name(im_name):
    return img_name_to_id[im_name]


def get_class_id(img_name):
    img_name, _ = img_name.split('.')
    img_name = img_name.rstrip('0123456789_0123456789')
    img_name = img_name.lower()
    return classNameToId[img_name]


def prepare_image(np_img):
    np_img = np.float32(np_img)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # convert to [C, H, W] from [H, W, C]
    np_img = np_img.transpose(2, 0, 1)
    channels, width, height = np_img.shape
    for channel in range(channels):
        np_img[channel] /= 255
        np_img[channel] -= mean[channel]
        np_img[channel] /= std[channel]

    return np_img


def tensor_to_image(img_tensor):
    img = img_tensor.numpy().transpose(1, 2, 0)
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


# code for gradient manipulation and iou

def check_overlap(pred, gt):
    x1_pred, y1_pred, x2_pred, y2_pred = pred
    x1_gt, y1_gt, x2_gt, y2_gt = gt
    return not (x2_pred <= x1_gt  # pred is to the left of gt
                or x1_pred >= x2_gt  # pred is to the right of gt
                or y2_pred <= y1_gt  # pred is above the gt
                or y1_pred >= y2_gt  # pred is below the gt
                )


def get_iou(pred, gt):
    x1_pred, y1_pred, x2_pred, y2_pred = pred
    x1_gt, y1_gt, x2_gt, y2_gt = gt
    x1_intersection = max(x1_pred, x1_gt)
    y1_intersection = max(y1_pred, y1_gt)
    x2_intersection = min(x2_pred, x2_gt)
    y2_intersection = min(y2_pred, y2_gt)

    intersection_area = (x2_intersection - x1_intersection + 1) * (y2_intersection - y1_intersection + 1)
    pred_area = (x2_pred - x1_pred + 1) * (y2_pred - y1_pred + 1)
    gt_area = (x2_gt - x1_gt + 1) * (y2_gt - y1_gt + 1)

    iou = float(intersection_area) / float(pred_area + gt_area - intersection_area)
    return iou


def bounding_box_grad(grad):
    # switch RGB -> BGR
    grad = grad[:, :, ::-1]
    # normalize the image to 0-255
    grad = cv2.normalize(grad, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    grad = grad.astype(np.uint8)

    im_gray = cv2.cvtColor(grad, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(im_gray, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    all_contours = []
    for j in range(len(contours)):
        all_contours.append(contours[j])

    all_contours = np.vstack(all_contours)
    pred_x, pred_y, pred_w, pred_h = cv2.boundingRect(all_contours)
    pred = [pred_x, pred_y, pred_x + pred_w, pred_y + pred_h]
    return pred


def load_model(checkpoint_path, num_labels, gpu_id):
    model = models.resnet50(pretrained=False)
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_labels)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    device = torch.device('cuda:' + gpu_id)
    model = model.to(device)
    model.eval()
    return model

