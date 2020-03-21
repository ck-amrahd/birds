import random


class BatchLoader:
    def __init__(self, train_folder_path, train_image_indices):

        self.train_image_indices = train_image_indices
        self.remaining_indices = train_image_indices
        self.train_folder_path = train_folder_path

    def update_remaining_indices(self, batch_indices):
        self.remaining_indices = list(set(self.remaining_indices) - set(batch_indices))

    def get_batch_indices(self, batch_size):

        if batch_size > len(self.remaining_indices):
            batch_indices = self.remaining_indices
        else:
            batch_indices = random.sample(self.remaining_indices, batch_size)

        self.update_remaining_indices(batch_indices)
        return batch_indices

    def reset(self):
        self.remaining_indices = self.train_image_indices
