import numpy as np
import torch
from PIL import Image
from torchvision import models
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

# try - grayscale [image x gradient]

img_path = 'viz_images/1.jpg'
checkpoint_path = '/home/user/Models/Experiment-4/Blackout/resnet50/pth_files/blackout_0.0_0.0.pth'
target_tensor = torch.tensor([1])
num_labels = 200

print('Running...')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = models.resnet50(pretrained=False)
input_features = model.fc.in_features
model.fc = nn.Linear(input_features, num_labels)

checkpoint = torch.load(checkpoint_path)
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
print(f'prediction: {prediction}')
# target_tensor = target_tensor.to(device)
model.zero_grad()
loss = criterion(output, target_tensor)
loss.backward()

# inp_grad = inputs.grad.data[3].numpy()
inputs_gradient = inputs.grad.data
# normalize the inputs_gradient
# inputs_gradient = torch.abs(inputs_gradient)
# inputs_gradient -= inputs_gradient.min()
# inputs_gradient /= inputs_gradient.max(
# print(f'inputs_gradient.shape: {inputs_gradient.shape}')

inp_grad = inputs_gradient[0]
# print(f'inp_grad.shape: {inp_grad.shape}')
inp_grad = np.abs(inp_grad)
inp_grad = inp_grad - inp_grad.min()
inp_grad /= inp_grad.max()
# transpose from [C, H, W] to [H, W, C]
grad = inp_grad.numpy().transpose(1, 2, 0)
grad_gray = np.dot(grad[..., :3], [0.2989, 0.5870, 0.1140])
# inp_grad *= 255
# inp_grad = inp_grad.astype(np.uint8)
# plt.imshow(grad)
# plt.show()

minVal1, maxVal1, minLoc1, maxLoc1 = cv2.minMaxLoc(grad_gray)
x1, y1 = maxLoc1

grad_gray[y1 - 5: y1 + 5, x1 - 5: x1 + 5] = 0
minVal2, maxVal2, minLoc2, maxLoc2 = cv2.minMaxLoc(grad_gray)
x2, y2 = maxLoc2

grad_gray[y2 - 5: y2 + 5, x2 - 5: x2 + 5] = 0
minVal3, maxVal3, minLoc3, maxLoc3 = cv2.minMaxLoc(grad_gray)
x3, y3 = maxLoc3

grad_gray[y3 - 5: y3 + 5, x3 - 5: x3 + 5] = 0
minVal4, maxVal4, minLoc4, maxLoc4 = cv2.minMaxLoc(grad_gray)
x4, y4 = maxLoc4

grad_gray[y4 - 5: y4 + 5, x4 - 5: x4 + 5] = 0
minVal5, maxVal5, minLoc5, maxLoc5 = cv2.minMaxLoc(grad_gray)
x5, y5 = maxLoc5

grad_gray[y5 - 5: y5 + 5, x5 - 5: x5 + 5] = 0
minVal6, maxVal6, minLoc6, maxLoc6 = cv2.minMaxLoc(grad_gray)
x6, y6 = maxLoc6

grad_gray[y6 - 5: y6 + 5, x6 - 5: x6 + 5] = 0
minVal7, maxVal7, minLoc7, maxLoc7 = cv2.minMaxLoc(grad_gray)
x7, y7 = maxLoc7

grad_gray[y7 - 5: y7 + 5, x7 - 5: x7 + 5] = 0
minVal8, maxVal8, minLoc8, maxLoc8 = cv2.minMaxLoc(grad_gray)
x8, y8 = maxLoc8

grad_gray[y8 - 5: y8 + 5, x8 - 5: x8 + 5] = 0
minVal9, maxVal9, minLoc9, maxLoc9 = cv2.minMaxLoc(grad_gray)
x9, y9 = maxLoc9

grad_gray[y9 - 5: y9 + 5, x9 - 5: x9 + 5] = 0
minVal10, maxVal10, minLoc10, maxLoc10 = cv2.minMaxLoc(grad_gray)
x10, y10 = maxLoc10


img = img.resize((224, 224), Image.ANTIALIAS)
# img = img.convert('L')
img = np.array(img, dtype=np.float32)
img /= img.max()
# grad_times_img = grad * img
# plt.imshow(grad_times_img)
cv2.circle(img, maxLoc1, 5, (0, 255, 0), 1)
cv2.circle(img, maxLoc2, 5, (0, 255, 0), 1)
cv2.circle(img, maxLoc3, 5, (0, 255, 0), 1)
cv2.circle(img, maxLoc4, 5, (0, 255, 0), 1)
cv2.circle(img, maxLoc5, 5, (0, 255, 0), 1)
cv2.circle(img, maxLoc6, 5, (0, 255, 0), 1)
cv2.circle(img, maxLoc7, 5, (0, 255, 0), 1)
cv2.circle(img, maxLoc8, 5, (0, 255, 0), 1)
cv2.circle(img, maxLoc9, 5, (0, 255, 0), 1)
cv2.circle(img, maxLoc10, 5, (0, 255, 0), 1)
plt.imshow(img)
plt.show()
# print('The model is giving importance at the following region: ')
# original_image = cv2.imread(image_path)
# original_image = original_image.resize((224, 224))
# np_img = np.asarray(original_image).astype('float32')
# # print(f'np_img.dtype: {np_img.dtype}')
# # print(f'inp_grad.dtype: {inp_grad.dtype}')
# img = 0.1 * np_img + 0.9 * inp_grad
# img = img.astype(np.uint8)
# img = Image.fromarray(img)
# plt.imshow(img, cmap='hot')
# plt.show()
