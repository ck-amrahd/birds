# This class will hold the required mappings from image id to image name and bounding box information
import cv2


class BoundingBox:
    def __init__(self, train_folder_path, img_file, bbox_file, height, width):
        self.train_folder_path = train_folder_path
        self.img_file = img_file
        self.bbox_file = bbox_file
        self.height = height
        self.width = width

        # image names will be in all lower()
        self.imgNameToId = {}
        # put the scaled x1, y1, x2, y2 values for bounding box
        self.idToBbox = {}

        with open(self.img_file, 'r') as read_file:
            for line in read_file:
                image_id, image_name = line.strip().split(' ')
                image_name = image_name.split('/')[-1].lower()
                self.imgNameToId[image_name] = image_id

        with open(self.bbox_file, 'r') as read_file:
            for line in read_file:
                image_id, x, y, width, height = line.strip().split(' ')
                self.idToBbox[image_id] = (x, y, width, height)

    def get_bbox(self, image_name):
        img = cv2.imread(self.train_folder_path + '/' + image_name)
        height, width, channels = img.shape
        image_id = self.imgNameToId[image_name.lower()]
        x, y, w, h = self.idToBbox[image_id]
        x, y, w, h = float(x), float(y), float(w), float(h)

        x_scale = self.width / width
        y_scale = self.height / height

        x_scaled = x * x_scale
        y_scaled = y * y_scale
        w_scaled = w * x_scale
        h_scaled = h * y_scale

        x1, y1, x2, y2 = int(x_scaled), int(y_scaled), int(x_scaled + w_scaled), int(y_scaled + h_scaled)

        return x1, y1, x2, y2
