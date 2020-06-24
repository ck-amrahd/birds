import torch
import torch.nn as nn
from torchvision import transforms
import foolbox as fb
import numpy as np
from torchvision import datasets, models
import pickle
import matplotlib.pyplot as plt

best_models_path = 'adversarial/best_models_exp7.pickle'
models_path = '/home/user/Models/Experiment-7/All'
test_robust_file = 'adversarial/test_robust_exp7.pickle'

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
                                          batch_size=64,
                                          shuffle=False,
                                          num_workers=16,
                                          pin_memory=True)

test_robust_acc = {}
best_models = pickle.load(open(best_models_path, 'rb'))

for model_class, models_list in best_models.items():
    test_robust_acc[model_class] = []

# loop for each value of epsilons
epsilons = np.linspace(0, 0.2, num=10)
print('Running...')
for idx, epsilon in enumerate(epsilons):
    for model_class, models_list in best_models.items():
        bounds = (0, 1)
        model_name = models_list[idx]
        model_path = models_path + '/' + model_name + '.pth'
        # model = models.resnet50(pretrained=False)
        model = models.resnet101(pretrained=False)
        input_features = model.fc.in_features
        model.fc = nn.Linear(input_features, num_classes)
        model.load_state_dict(torch.load(model_path))

        model.eval()
        fmodel = fb.PyTorchModel(model, bounds=bounds)
        attack = fb.attacks.FGSM()

        robust_acc_list = []
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, _, is_adv = attack(fmodel, inputs, labels, epsilons=epsilon)
            robust_acc = 1 - is_adv.float().mean()
            robust_acc_list.append(robust_acc.cpu().numpy())

        avg_acc = np.mean(robust_acc_list)
        test_robust_acc[model_class].append(avg_acc)

# save the test_robust_acc
with open(test_robust_file, 'wb') as write_file:
    pickle.dump(test_robust_acc, write_file)


# fig = plt.figure(figsize=(19.20, 10.80))
for model_class, robust_acc in test_robust_acc.items():
    plt.plot(epsilons, robust_acc, label=model_class)

# plt.savefig('test.png', format='png')
plt.legend()
plt.show()
