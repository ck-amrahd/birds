from torchvision import models
import torch.nn as nn
import torch
import torch.optim as optim
import os
from batch_loader import BatchLoader
from torchvision import transforms, datasets


class Model:
    def __init__(self, model_name, train_folder_path, test_folder_path, x_train, y_train, device, num_channels, height,
                 width, checkpoint_path, bounding_box=None, segmentation_mask=None, num_labels=200):
        """
        Initializes the model along with other initialization
        """

        self.model_name = model_name
        self.train_folder_path = train_folder_path
        self.test_folder_path = test_folder_path
        self.x_train = x_train
        self.y_train = y_train
        self.bounding_box = bounding_box
        self.segmentation_mask = segmentation_mask
        self.device = device
        self.num_labels = num_labels
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.checkpoint_path = checkpoint_path
        self.train_dataset_length = len(os.listdir(self.train_folder_path))

        self.test_dataset_length = 0
        test_subfolders = os.listdir(self.test_folder_path)
        for item in test_subfolders:
            self.test_dataset_length += len(os.listdir(self.test_folder_path + '/' + item))

        # define test loaders
        self.test_transform = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.test_dataset = datasets.ImageFolder(self.test_folder_path, self.test_transform)

        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                       batch_size=32,
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

        elif self.model_name == 'inception_v3':
            model = models.inception_v3(pretrained=False)
            model.fc = nn.Linear(2048, 8142)
            model.aux_logits = False
            model = model.to(self.device)
            checkpoint = torch.load('./inception_v3.pth.tar')
            model.load_state_dict(checkpoint['state_dict'])
            model.fc = nn.Linear(2048, self.num_labels)

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

    def calculate_penalty_mask(self, batch_indices, input_gradient):
        batch_size = len(batch_indices)

        # make inside mask to 1 and outside mask to zeros, so when we take element-wise product with the
        # input gradient, we will just get a patch from inside the mask

        penalty_inside_box = torch.zeros(batch_size, self.num_channels, self.height, self.width)
        for index, item in enumerate(batch_indices):
            mask_tensor = self.segmentation_mask[item]
            temp_grad = penalty_inside_box[index]
            temp_grad[mask_tensor == 255.0] = 1.0
            penalty_inside_box[index] = temp_grad

        penalty_inside_box = penalty_inside_box.to(self.device)
        penalty_inside_box = penalty_inside_box * input_gradient
        penalty_inside_box = (torch.norm(penalty_inside_box)) ** 2

        # make inside mask to 0 and outside mask to ones, so when we take element-wise product with the
        # input gradient, we will just get a patch outside of the mask

        penalty_outside_box = torch.ones(batch_size, self.num_channels, self.height, self.width)
        for index, item in enumerate(batch_indices):
            mask_tensor = self.segmentation_mask[item]
            temp_grad = penalty_outside_box[index]
            temp_grad[mask_tensor == 255.0] = 0.0
            penalty_outside_box[index] = temp_grad

        penalty_outside_box = penalty_outside_box.to(self.device)
        penalty_outside_box = penalty_outside_box * input_gradient
        penalty_outside_box = (torch.norm(penalty_outside_box)) ** 2

        return penalty_inside_box, penalty_outside_box

    def train(self, train_image_indices, batch_size, train_acc_list, test_acc_list, train_loss_list, test_loss_list,
              num_epochs=50, train_with_bbox=False, train_with_seg_mask=False, lambda_1=0, lambda_2=0,
              start_from_pretrained_model=True, learning_rate=0.01, optimizer='SGD', best_acc=0.0):

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
        best_acc_this_run = 0.0

        for epoch in range(num_epochs):
            model.train()
            train_batch_loader.reset()
            print('Epoch: {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 15)

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

                if train_with_bbox:
                    inputs.requires_grad_()
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    loss = criterion(outputs, labels)
                    input_gradient = torch.autograd.grad(loss, inputs, create_graph=True)[0]
                    penalty_inside_box, penalty_outside_box = self.calculate_penalty_box(batch_indices, input_gradient)
                    new_loss = loss + lambda_1 * penalty_inside_box + lambda_2 * penalty_outside_box
                    new_loss.backward()
                    optimizer.step()

                elif train_with_seg_mask:
                    inputs.requires_grad_()
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    loss = criterion(outputs, labels)
                    input_gradient = torch.autograd.grad(loss, inputs, create_graph=True)[0]
                    penalty_inside_box, penalty_outside_box = self.calculate_penalty_mask(batch_indices, input_gradient)
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
            test_correct = 0.0
            test_loss = 0.0

            model.eval()
            with torch.no_grad():
                for inputs_test, labels_test in self.test_loader:
                    inputs_test, labels_test = inputs_test.to(self.device), labels_test.to(self.device)
                    outputs_test = model(inputs_test)
                    preds_test = torch.argmax(outputs_test, dim=1)
                    loss_test = criterion(outputs_test, labels_test)

                    test_loss += loss_test.item()
                    test_correct += torch.sum(preds_test == labels_test).float().item()

            test_loss = test_loss / self.test_dataset_length
            test_loss_list.append(test_loss)
            test_acc = test_correct / self.test_dataset_length
            test_acc *= 100.0
            test_acc_list.append(test_acc)
            print('test Loss: {:.4f} Acc: {:.4f} % \n'.format(test_loss, test_acc))

            if test_acc > best_acc:
                best_acc = test_acc
                best_model = model.state_dict()
                if os.path.exists(self.checkpoint_path):
                    os.remove(self.checkpoint_path)

                torch.save(best_model, self.checkpoint_path)

            if test_acc > best_acc_this_run:
                best_acc_this_run = test_acc

        # model.load_state_dict(best_model)
        return best_acc_this_run, penalty_inside_list, penalty_outside_list
