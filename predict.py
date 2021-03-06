# predict and return predictions, gradients
import numpy as np
import torch
from PIL import Image


def predict(model, img_path, transform, criterion, target_tensor, height, width, num_channels, gpu_id):
    device = torch.device('cuda:' + gpu_id)
    inputs = torch.zeros(1, num_channels, height, width)
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')
    img_tensor = transform(img)
    inputs[0] = img_tensor

    inputs, target_tensor = inputs.to(device), target_tensor.to(device)
    inputs.requires_grad_()
    output = model(inputs)
    prediction = torch.argmax(output, dim=1).item()
    model.zero_grad()
    loss = criterion(output, target_tensor)
    loss.backward()

    inputs_gradient = inputs.grad.data
    inp_grad = inputs_gradient[0]

    grad = inp_grad.cpu().numpy().transpose(1, 2, 0)
    # grad = np.max(grad, axis=2)
    grad = np.abs(grad)
    grad = grad - grad.min()
    grad /= grad.max()

    return prediction, grad
