import torch
import numpy as np
import torch
from PIL import Image
from torchvision import models
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from utils import tensor_to_image

image_folder = 'viz_images'
class_id = 3
total_images = 4
viz_index = 3
checkpoint_path = 'results/foolbox3/model_bbox_0.0_100.0.pth'
target_tensor = torch.tensor([0, 1, 2, 3])
num_labels = 200

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
inputs = torch.zeros(total_images, 3, 224, 224)

for idx in range(4):
    img = Image.open(image_folder + '/' + str(idx) + '.jpg')
    if img.mode == 'L':
        img = img.convert('RGB')
    img_tensor = transform(img)
    inputs[idx] = img_tensor

inputs.requires_grad_()
output = model(inputs)
prediction = torch.argmax(output, dim=1)
print(f'prediction: {prediction}')
# target_tensor = target_tensor.to(device)
loss = criterion(output, target_tensor)
loss.backward()

# inp_grad = inputs.grad.data[3].numpy()
inputs_gradient = inputs.grad.data
# normalize the inputs_gradient
# inputs_gradient = torch.abs(inputs_gradient)
# inputs_gradient -= inputs_gradient.min()
# inputs_gradient /= inputs_gradient.max(
# print(f'inputs_gradient.shape: {inputs_gradient.shape}')

inp_grad = inputs_gradient[viz_index]
# print(f'inp_grad.shape: {inp_grad.shape}')
inp_grad = np.abs(inp_grad)
inp_grad = inp_grad - inp_grad.min()
inp_grad /= inp_grad.max()
# transpose from [C, H, W] to [H, W, C]
# inp_grad = inp_grad.transpose(1, 2, 0)
# inp_grad = np.dot(inp_grad[...,:3], [0.2989, 0.5870, 0.1140])
# inp_grad *= 255
# inp_grad = inp_grad.astype(np.uint8)
img = inp_grad.numpy().transpose(1, 2, 0)
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
