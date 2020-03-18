# copy train and test images to different folder.

import os
import shutil

image_id_to_name = 'data/images.txt'
# create imageIdName dictionary that will have image_id as the key and the image_name as the value
imageIdName = {}
with open(image_id_to_name, 'r') as read_file:
    for line in read_file:
        line = line.strip()
        image_id, image_name = line.split(' ')
        imageIdName[image_id] = image_name

train_test_file = 'data/train_test_split.txt'
# create dictionary that will hold image_id and is_train to later move them to separate folders
imageIdIsTrain = {}
with open(train_test_file, 'r') as read_file:
    for line in read_file:
        line = line.strip()
        image_id, is_train_image = line.split(' ')
        imageIdIsTrain[image_id] = is_train_image

# create a dictionary that holds class id to class name
classIdToName = {}
class_id_to_class_name = 'data/classes.txt'
with open(class_id_to_class_name, 'r') as read_file:
    for line in read_file:
        line = line.strip()
        class_id, class_name = line.split(' ')
        classIdToName[class_id] = class_name

train_folder = 'data/train'
test_folder = 'data/test'

if os.path.exists(train_folder):
    shutil.rmtree(train_folder)

os.makedirs(train_folder)

if os.path.exists(test_folder):
    shutil.rmtree(test_folder)

os.makedirs(test_folder)
# create subfolders inside test folder to put into correct classes -- create 200 subfolders
for key, value in classIdToName.items():
    os.makedirs(test_folder + '/' + value)
    # os.makedirs(train_folder + '/' + value)

for key, value in imageIdIsTrain.items():
    image_name = imageIdName[key]
    folder_name, file_name = image_name.strip().split('/')
    source_file = 'data/images/' + folder_name + '/' + file_name
    if value == '0':
        # These are test images - move them to specific folders
        shutil.copy(source_file, test_folder + '/' + folder_name + '/' + file_name)
    else:
        # value - 1 corresponds to training images, move them to specific folders
        shutil.copy(source_file, train_folder + '/' + file_name)
