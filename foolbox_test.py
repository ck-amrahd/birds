import torch
import torch.nn as nn
from torchvision import transforms
import foolbox as fb
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, models
import os

models_folder = 'results/resnet50'

test_dataset_path = 'data/test'
num_classes = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(test_dataset_path, transform)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=32,
                                          shuffle=False,
                                          num_workers=16,
                                          pin_memory=True)

models_list = os.listdir(models_folder)
bounds = (0, 1)
epsilons = np.linspace(0, 0.1, num=20)
info = {}
print(f'Running Attacks...')
for model_name in models_list:
    model = models.resnet50(pretrained=False)
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)
    model.load_state_dict(torch.load(models_folder + '/' + model_name))

    model.eval()
    fmodel = fb.PyTorchModel(model, bounds=bounds)
    attack = fb.attacks.FGSM()

    robust_acc_list = []
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        _, _, is_adv = attack(fmodel, inputs, labels, epsilons=epsilons)
        robust_acc = 1 - is_adv.float().mean(axis=-1)
        robust_acc_list.append(robust_acc)

    robust_acc = torch.stack(robust_acc_list)
    robust_acc = torch.mean(robust_acc, dim=0)
    model_name = model_name.strip().split('.')[0]
    info[model_name] = robust_acc

for model_name, robust_acc in info.items():
    plt.plot(epsilons, robust_acc.cpu().numpy(), label=model_name)

plt.legend()
plt.show()
