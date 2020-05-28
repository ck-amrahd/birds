import torch
import torch.nn as nn
from torchvision import transforms
import foolbox as fb
import numpy as np
from torchvision import datasets, models
import os
import pickle
import time

start = time.time()

# This will generate all_info.pickle file by running attacks against the validation set
# Then use info.py to select best models from each class of models
# Then use foolbox_test.py to run the best models against the test set

# We need to generate 5 curves
# one for normal, another for blackout, another for lamda_1=0 and varying lambda_2
# another for lambda_1=lambda_2 and another for varying lambda_1 and lambda_2

log_path = 'adversarial/all_info_exp3.pickle'
experiment_folder = '/home/user/Models/Experiment-3/'

# put lambda_1_zero into lambda_vary and create four classes
# 'lambda_1_zero': experiment_folder + 'BboxL10/resnet50/pth_files',

model_folders = {'normal': experiment_folder + 'Normal/resnet50/pth_files',
                 'blackout': experiment_folder + 'Blackout/resnet50/pth_files',
                 'lambda_equal': experiment_folder + 'BboxEqualL1L2/resnet50/pth_files',
                 'lambda_vary': experiment_folder + 'BboxL1L2/resnet50/pth_files'}

# We will use val set to select the best values of lambda_1 and lambda_2 from each family of the above
# discussed models.

val_dataset_path = 'data/val'
num_classes = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(val_dataset_path, transform)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=64,
                                         shuffle=False,
                                         num_workers=16,
                                         pin_memory=True)

all_info = {}
for train_method, folder_path in model_folders.items():
    models_list = os.listdir(folder_path)
    bounds = (0, 1)
    epsilons = np.linspace(0, 0.2, num=20)
    info = {}
    print(f'Running Attacks...')
    for model_name in models_list:
        model = models.resnet50(pretrained=False)
        input_features = model.fc.in_features
        model.fc = nn.Linear(input_features, num_classes)
        model.load_state_dict(torch.load(folder_path + '/' + model_name))

        model.eval()
        fmodel = fb.PyTorchModel(model, bounds=bounds)
        attack = fb.attacks.FGSM()

        robust_acc_list = []
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, _, is_adv = attack(fmodel, inputs, labels, epsilons=epsilons)
            robust_acc = 1 - is_adv.float().mean(axis=-1)
            robust_acc_list.append(robust_acc)

        robust_acc = torch.stack(robust_acc_list)
        robust_acc = torch.mean(robust_acc, dim=0)
        model_name = model_name.strip().rsplit('.', 1)[0]
        info[model_name] = robust_acc

        # for model, robust_acc in info.items():
        #    plt.plot(epsilons, robust_acc.cpu().numpy(), label=model)

        # plt.legend()
        # plt.show()

    all_info[train_method] = info

with open(log_path, 'wb') as write_file:
    pickle.dump(all_info, write_file)

end = time.time()

elapsed_hours = (end - start) / (60 * 60)
print(f'elapsed_hours: {elapsed_hours}')
