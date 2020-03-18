# segmentation class that will hold the segmentation mask for each image

import torch
import numpy as np
import cv2


class Segmentation:
    def __init__(self, train_folder, img_file, seg_folder, height, width):
        self.train_folder = train_folder
        self.img_file = img_file
        self.seg_folder = seg_folder
        self.height = height
        self.width = width

        self.nameToId = {}
        self.idToSegMask = {}

        with open(self.img_file, 'r') as read_file:
            for line in read_file:
                img_id, img_name = line.strip().split(' ')
                seg_mask_name = img_name.rsplit('.', 1)[0] + '.png'
                sef_mask_path = self.seg_folder + '/' + seg_mask_name
                self.idToSegMask[img_id] = sef_mask_path
                img_name = img_name.split('/')[-1].lower()
                self.nameToId[img_name] = img_id

    def get_segmentation_mask_path(self, img_name):
        img_id = self.nameToId[img_name.lower()]
        sef_mask_path = self.idToSegMask[img_id]
        return sef_mask_path

    def get_segmentation_mask(self, img_name):
        # returns mask of shape [3, 224, 224] with 255.0.0 on the places of mask and 0 elsewhere
        mask_path = self.get_segmentation_mask_path(img_name)
        mask = cv2.imread(mask_path)
        mask[mask != 0] = 255.0
        mask[mask == 0] = 0.0
        mask = cv2.resize(mask, (self.height, self.width))
        mask = np.transpose(mask, (2, 0, 1)).astype('float32')
        mask_tensor = torch.from_numpy(mask)
        return mask_tensor
