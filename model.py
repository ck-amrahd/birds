from torchvision import models
import torch.nn as nn
import torch
import torch.optim as optim
import os
from batch_loader import BatchLoader
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import StepLR


class Model:
    def __init__(self, model_name, train_folder_path, x_train, y_train, val_path, device, num_channels, height,
                 width, checkpoint_path, bounding_box=None, num_labels=200):
        """
        Initializes the model along with other initialization
        """

        self.model_name = model_name
        self.train_folder_path = train_folder_path
        self.x_train = x_train
        self.y_train = y_train
        self.bounding_box = bounding_box
        self.val_path = val_path
        self.device = device
        self.num_labels = num_labels
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.checkpoint_path = checkpoint_path
        self.train_dataset_length = len(os.listdir(self.train_folder_path))

        self.val_dataset_length = 0
        val_subfolders = os.listdir(self.val_path)
        for item in val_subfolders:
            self.val_dataset_length += len(os.listdir(self.val_path + '/' + item))

        # define test loaders
        self.val_transform = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.val_dataset = datasets.ImageFolder(self.val_path, self.val_transform)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset,
                                                      batch_size=64,
                                                      shuffle=False,
                                                      num_workers=16,
                                                      pin_memory=True)

    def initialize_model(self, start_from_pretrained_model=True):
        if self.model_name == 'alexnet':
            if start_from_pretrained_model:
                model = models.alexnet(pretrained=True)
            else:
                model = models.alexnet(pretrained=False)
            input_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(input_features, self.num_labels)

        elif self.model_name == 'vgg19':
            if start_from_pretrained_model:
                model = models.vgg19(pretrained=True)
            else:
                model = models.vgg19(pretrained=False)
            input_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(input_features, self.num_labels)

        elif self.model_name == 'vgg16':
            if start_from_pretrained_model:
                model = models.vgg16(pretrained=True)
            else:
                model = models.vgg16(pretrained=False)
            input_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(input_features, self.num_labels)

        elif self.model_name == 'resnet152':
            if start_from_pretrained_model:
                model = models.resnet152(pretrained=True)
            else:
                model = models.resnet152(pretrained=False)
            input_features = model.fc.in_features
            model.fc = nn.Linear(input_features, self.num_labels)

        elif self.model_name == 'resnet50':
            if start_from_pretrained_model:
                model = models.resnet50(pretrained=True)
            else:
                model = models.resnet50(pretrained=False)
            input_features = model.fc.in_features
            model.fc = nn.Linear(input_features, self.num_labels)

        else:
            if start_from_pretrained_model:
                model = models.resnet18(pretrained=True)
            else:
                model = models.resnet18(pretrained=True)
            input_features = model.fc.in_features
            model.fc = nn.Linear(input_features, self.num_labels)

        return model

    def calculate_penalty_box(self, batch_indices, input_gradient):
        batch_size = len(batch_indices)

        # make inside box region to 1 and outside box to zeros, so when we take element-wise product with the
        # input gradient, we will just get a patch from inside the box

        penalty_inside_box = torch.zeros(batch_size, self.num_channels, self.height, self.width)
        for index, item in enumerate(batch_indices):
            x1, y1, x2, y2 = self.bounding_box[item]
            penalty_inside_box[index, :, y1:y2, x1:x2] = 1.0

        penalty_inside_box = penalty_inside_box.to(self.device)
        penalty_inside_box = penalty_inside_box * input_gradient
        penalty_inside_box = (torch.norm(penalty_inside_box)) ** 2

        # per example norm calculation
        # penalty_inside_box = penalty_inside_box.view(batch_size, -1)
        # penalty_inside_box = torch.sum(penalty_inside_box ** 2, dim=1)
        # penalty_inside_box = penalty_inside_box.sum()

        # make inside box to 0 and outside box to ones, so when we take element-wise product with the
        # input gradient, we will just get a patch from outside the box

        penalty_outside_box = torch.ones(batch_size, self.num_channels, self.height, self.width)
        for index, item in enumerate(batch_indices):
            x1, y1, x2, y2 = self.bounding_box[item]
            penalty_outside_box[index, :, y1:y2, x1:x2] = 0.0

        penalty_outside_box = penalty_outside_box.to(self.device)
        penalty_outside_box = penalty_outside_box * input_gradient
        penalty_outside_box = (torch.norm(penalty_outside_box)) ** 2

        return penalty_inside_box, penalty_outside_box

    def train(self, train_image_indices, batch_size, num_epochs=50, train_method='normal', lambda_1=0, lambda_2=0,
              start_from_pretrained_model=True, learning_rate=0.01, optimizer='SGD'):

        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)

        model = self.initialize_model(start_from_pretrained_model=start_from_pretrained_model)

        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()

        if optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

        elif optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

        else:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

        train_batch_loader = BatchLoader(self.train_folder_path, train_image_indices)

        n_images = len(train_image_indices)
        if n_images % batch_size == 0:
            num_batches = n_images // batch_size
        else:
            num_batches = (n_images // batch_size) + 1

        penalty_inside_list = []
        penalty_outside_list = []
        train_acc_list = []
        train_loss_list = []
        val_loss_list = []
        val_acc_list = []
        best_acc = 0.0

        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        for epoch in range(num_epochs):
            model.train()
            train_batch_loader.reset()
            print('Epoch: {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 50)

            train_correct = 0.0
            train_loss = 0.0
            penalty_inside = 0.0
            penalty_outside = 0.0

            for batch in range(num_batches):
                batch_indices = train_batch_loader.get_batch_indices(batch_size)
                inputs = self.x_train[batch_indices]
                labels = self.y_train[batch_indices]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                if train_method == 'bbox':
                    inputs.requires_grad_()
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    loss = criterion(outputs, labels)
                    input_gradient = torch.autograd.grad(loss, inputs, create_graph=True)[0]
                    penalty_inside_box, penalty_outside_box = self.calculate_penalty_box(batch_indices, input_gradient)
                    new_loss = loss + lambda_1 * penalty_inside_box + lambda_2 * penalty_outside_box
                    new_loss.backward()
                    optimizer.step()

                else:
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    penalty_inside_box = torch.tensor(0).to(self.device)
                    penalty_outside_box = torch.tensor(0).to(self.device)

                train_loss += loss.item()
                train_correct += torch.sum(preds == labels).float().item()
                penalty_inside += penalty_inside_box.item() * lambda_1
                penalty_outside += penalty_outside_box.item() * lambda_2

            train_loss = train_loss / self.train_dataset_length
            train_loss_list.append(train_loss)
            train_acc = (train_correct / self.train_dataset_length) * 100.0
            train_acc_list.append(train_acc)
            penalty_inside = penalty_inside / self.train_dataset_length
            penalty_outside = penalty_outside / self.train_dataset_length
            penalty_inside_list.append(penalty_inside)
            penalty_outside_list.append(penalty_outside)

            print('Train Loss: {:.4f} Acc: {:.4f} % '.format(train_loss, train_acc))
            print(f'Penalty Inside Box: {round(penalty_inside, 4)}')
            print(f'Penalty Outside Box: {round(penalty_outside, 4)}')

            # validate after each epoch
            val_correct = 0.0
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for inputs_val, labels_val in self.val_loader:
                    inputs_val, labels_val = inputs_val.to(self.device), labels_val.to(self.device)
                    outputs_val = model(inputs_val)
                    preds_val = torch.argmax(outputs_val, dim=1)
                    loss_test = criterion(outputs_val, labels_val)

                    val_loss += loss_test.item()
                    val_correct += torch.sum(preds_val == labels_val).float().item()

            val_loss = val_loss / self.val_dataset_length
            val_loss_list.append(val_loss)
            val_acc = (val_correct / self.val_dataset_length) * 100.0
            val_acc_list.append(val_acc)
            print('Val Loss: {:.4f} Acc: {:.4f} % \n'.format(val_loss, val_acc))

            # save the best model
            if val_acc > best_acc:
                best_acc = val_acc
                model.state_dict()
                if os.path.exists(self.checkpoint_path):
                    os.remove(self.checkpoint_path)

                torch.save(model.state_dict(), self.checkpoint_path)

            scheduler.step()

        return_dict = {'train_acc_list': train_acc_list,
                       'train_loss_list': train_loss_list,
                       'penalty_inside_list': penalty_inside_list,
                       'penalty_outside_list': penalty_outside_list,
                       'val_loss_list': val_loss_list,
                       'val_acc_list': val_acc_list,
                       'best_acc': best_acc}

        return return_dict
