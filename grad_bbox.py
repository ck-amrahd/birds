import numpy as np
import torch
from PIL import Image
from torchvision import models
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
from bounding_box import BoundingBox

test_folder_path = 'data/test'
images_text_file = 'data/images.txt'
bounding_box_file = 'data/bounding_boxes.txt'
height = 224
width = 224

img_path = 'data/test/002.Laysan_Albatross/Laysan_Albatross_0033_658.jpg'
checkpoint_path = '/home/user/Models/Experiment-4/BboxL1L2/resnet50/pth_files/bbox_1000.0_21.54.pth'
# checkpoint_path = '/home/user/Models/Experiment-4/BboxL1L2/resnet50/pth_files/bbox_0.0_100.0.pth'
# checkpoint_path = '/home/user/Models/Experiment-4/BboxEqualL1L2/resnet50/pth_files/bbox_1000.0_1000.0.pth'
# checkpoint_path = '/home/user/Models/Experiment-4/Normal/resnet50/pth_files/normal_0.0_0.0.pth'

target_tensor = torch.tensor([1])
num_labels = 200

print('Running...')
transform = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

bbox = BoundingBox(test_folder_path, images_text_file, bounding_box_file, height, width)
# show image and ground-truth bounding box
x1_gt, y1_gt, x2_gt, y2_gt = bbox.get_bbox_from_path(img_path)
orig_img = cv2.imread(img_path)
orig_img = cv2.resize(orig_img, (height, width))
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
cv2.rectangle(orig_img, (int(x1_gt), int(y1_gt)), (int(x2_gt), int(y2_gt)), (0, 255, 0), 2)


model = models.resnet50(pretrained=False)
input_features = model.fc.in_features
model.fc = nn.Linear(input_features, num_labels)

checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

criterion = nn.CrossEntropyLoss()
inputs = torch.zeros(1, 3, 224, 224)
img = Image.open(img_path)
if img.mode == 'L':
    img = img.convert('RGB')
img_tensor = transform(img)
inputs[0] = img_tensor

inputs.requires_grad_()
output = model(inputs)
prediction = torch.argmax(output, dim=1)

# print(f'label: {target_tensor.item()}')
# print(f'prediction: {prediction.item()}')

model.zero_grad()
loss = criterion(output, target_tensor)
loss.backward()

inputs_gradient = inputs.grad.data
inp_grad = inputs_gradient[0]

grad = inp_grad.numpy().transpose(1, 2, 0)
# grad = np.max(grad, axis=2)
grad = np.abs(grad)
grad = grad - grad.min()
grad /= grad.max()

# switch RGB -> BGR
grad = grad[:, :, ::-1]
# normalize the image to 0-255
grad = cv2.normalize(grad, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
grad = grad.astype(np.uint8)

imGray = cv2.cvtColor(grad, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imGray, 50, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

all_contours = []
for i in range(len(contours)):
    all_contours.append(contours[i])

all_contours = np.vstack(all_contours)

pred_x, pred_y, pred_w, pred_h = cv2.boundingRect(all_contours)
x1_pred = pred_x
y1_pred = pred_y
x2_pred = pred_x + pred_w
y2_pred = pred_y + pred_h

cv2.rectangle(orig_img, (x1_pred, y1_pred), (x2_pred, y2_pred), (255, 0, 0), 2)

# check if overlap
overlap = not(x2_pred <= x1_gt      # pred is to the left of gt
              or x1_pred >= x2_gt   # pred is to the right of gt
              or y2_pred <= y1_gt   # pred is above the gt
              or y1_pred >= y2_gt   # pred is below the gt
              )

print(f'overlap: {overlap}')

# calculate iou in case of overlap else the iou is 0

if overlap:
    x1_intersection = max(x1_pred, x1_gt)
    y1_intersection = max(y1_pred, y1_gt)
    x2_intersection = min(x2_pred, x2_gt)
    y2_intersection = min(y2_pred, y2_gt)

    intersection_area = (x2_intersection - x1_intersection + 1) * (y2_intersection - y1_intersection + 1)
    pred_area = (x2_pred - x1_pred + 1) * (y2_pred - y1_pred + 1)
    gt_area = (x2_gt - x1_gt + 1) * (y2_gt - y1_gt + 1)

    iou = float(intersection_area) / float(pred_area + gt_area - intersection_area)

else:
    iou = 0.0

print(f'iou: {round(iou, 2)}')
# calculate saliency score

plt.imshow(orig_img)
plt.show()
