# move half images from val folder to test folder
import os
import shutil

source_folder = 'data/original_test'
test_folder = 'data/test'
val_folder = 'data/val'

if os.path.exists(test_folder):
    shutil.rmtree(test_folder)

os.makedirs(test_folder)

if os.path.exists(val_folder):
    shutil.rmtree(val_folder)

os.makedirs(val_folder)

for item in os.listdir(source_folder):
    os.makedirs(test_folder + '/' + item)
    os.makedirs(val_folder + '/' + item)
    temp_file_names = []
    for file_names in os.listdir(source_folder + '/' + item):
        temp_file_names.append(file_names)

    num_images = len(temp_file_names)
    middle_value = num_images // 2
    val_images = temp_file_names[0:middle_value]
    test_images = temp_file_names[middle_value:]

    for img_name in val_images:
        source_file = source_folder + '/' + item + '/' + img_name
        destination_file = val_folder + '/' + item + '/' + img_name
        shutil.copy(source_file, destination_file)

    for img_name in test_images:
        source_file = source_folder + '/' + item + '/' + img_name
        destination_file = test_folder + '/' + item + '/' + img_name
        shutil.copy(source_file, destination_file)