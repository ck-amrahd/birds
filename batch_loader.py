import random
from torchvision import transforms


class BatchLoader:
    def __init__(self, train_folder_path, num_channels, height, width, train_image_indices):

        self.train_image_indices = train_image_indices
        self.remaining_indices = train_image_indices
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.train_folder_path = train_folder_path

        self.transform = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def update_remaining_indices(self, batch_indices):
        self.remaining_indices = list(set(self.remaining_indices) - set(batch_indices))

    # now batch_loader will only return the indices not the inputs and labels anymore
    def get_batch_indices(self, batch_size):
        # will return a batch size and keep track of the indices
        # after returning a batch update the remaining_indices
        # The problem may arise if the batch_size > remaining_indices
        # in such case randomly take out some images from training indices to fill out the rest

        if batch_size > len(self.remaining_indices):
            batch_indices = self.remaining_indices
        else:
            batch_indices = random.sample(self.remaining_indices, batch_size)

        self.update_remaining_indices(batch_indices)
        return batch_indices

    def reset(self):
        self.remaining_indices = self.train_image_indices
