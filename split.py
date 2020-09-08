# data/images has all the images
# we need to split 1/2 for training, 1/6 for train_val, 1/6 for val and 1/6 for test
# train_val is used to select best model during training
# val is used to select lambda_1 and lambda_2
# test is the final test set for robust accuracy testing

import os
import shutil
import random
import math

images_path = 'data/images'
train_path = 'data/train'
train_val_path = 'data/train_val'
val_path = 'data/val'
test_path = 'data/test'

print('Running...')

if os.path.exists(train_path):
    shutil.rmtree(train_path)
os.mkdir(train_path)

if os.path.exists(train_val_path):
    shutil.rmtree(train_val_path)
os.mkdir(train_val_path)

if os.path.exists(val_path):
    shutil.rmtree(val_path)
os.mkdir(val_path)

if os.path.exists(test_path):
    shutil.rmtree(test_path)
os.mkdir(test_path)

folders = sorted(os.listdir(images_path))

total_train_images = 0
total_train_val_images = 0
total_val_images = 0
total_test_images = 0

train_counter = 0
train_val_counter = 0
val_counter = 0
test_counter = 0


for folder_name in folders:

    # for training images let's put everything in one place so that we can load them to RAM at first
    # And we don't have to load everytime during training which helps to train faster also it's easy
    # to load the bounding box at the start-up
    # os.mkdir(train_path + '/' + folder_name)
    # for others let's create sub-folders

    os.mkdir(train_val_path + '/' + folder_name)
    os.mkdir(val_path + '/' + folder_name)
    os.mkdir(test_path + '/' + folder_name)

    images = os.listdir(images_path + '/' + folder_name)
    total_images = len(images)
    indices = list(range(total_images))
    train_indices = random.sample(indices, math.ceil(total_images / 2))
    train_images = []
    for idx in train_indices:
        train_images.append(images[idx])

    total_train_images += len(train_images)
    # now create new list by removing the training images
    images = [j for i, j in enumerate(images) if i not in train_indices]

    # check everything is correct
    for img_name in train_images:
        if img_name in images:
            print('Something wrong!!!')
            exit()

    total_images = len(images)
    # because I sampled training images randomly, remaining can be deterministic
    if total_images % 3 == 0:
        split_index = int(total_images / 3)
    else:
        split_index = math.ceil(total_images / 3)

    train_val_images = images[0:split_index]
    total_train_val_images += len(train_val_images)

    val_images = images[split_index: 2 * split_index]
    total_val_images += len(val_images)

    test_images = images[2 * split_index:]
    total_test_images += len(test_images)

    # check the test images are at least 5/class
    if len(test_images) < 5:
        print(f'few test examples in: {folder_name}')

    # copy images to respective folders for all train, train_val, val and test

    # copy train images
    for img_name in train_images[::2]:
        source_file = images_path + '/' + folder_name + '/' + img_name
        destination_file = train_path + '/' + img_name
        shutil.copy(source_file, destination_file)
        train_counter += 1

    # copy train_val images
    for img_name in train_val_images[::2]:
        source_file = images_path + '/' + folder_name + '/' + img_name
        destination_file = train_val_path + '/' + folder_name + '/' + img_name
        shutil.copy(source_file, destination_file)
        train_val_counter += 1

    # copy val images
    for img_name in val_images[::2]:
        source_file = images_path + '/' + folder_name + '/' + img_name
        destination_file = val_path + '/' + folder_name + '/' + img_name
        shutil.copy(source_file, destination_file)
        val_counter += 1

    # copy test images
    for img_name in test_images[::2]:
        source_file = images_path + '/' + folder_name + '/' + img_name
        destination_file = test_path + '/' + folder_name + '/' + img_name
        shutil.copy(source_file, destination_file)
        test_counter += 1

print(f'total_train_images: {total_train_images}')
print(f'total_train_val_images: {total_train_val_images}')
print(f'total_val_images: {total_val_images}')
print(f'total_test_images: {total_test_images}')

print(f'total_train_images_copied: {train_counter}')
print(f'total_train_val_images_copied: {train_val_counter}')
print(f'total_val_images_copied: {val_counter}')
print(f'total_test_images_copied: {test_counter}')

print('Done')
