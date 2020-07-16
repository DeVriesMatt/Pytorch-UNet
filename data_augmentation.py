import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython import display
import time
import glob

from skimage.transform import rotate
from skimage.util import random_noise
from skimage.filters import gaussian
from scipy import ndimage
import numpy as np
from tqdm import tqdm
import imageio
from util import create_dir_if_not_exist


PATCH_GT = "./processed/IOSTAR/processed/GT/patch_128/images"
PATCH_IMAGE = "./processed/IOSTAR/processed/image/patch_128/images"
AUG_PROC_GT = "./augmented_patch/GT/"
AUG_PROC_IMAGE = "./augmented_patch/image/"

images = []

for img_path in sorted(glob.glob(PATCH_IMAGE + '/*.png')):
    # print(img_path)
    images.append(mpimg.imread(img_path, 0))

ground_truth = []

for img_path in sorted(glob.glob(PATCH_GT + '/*.png')):
    # print(img_path)
    ground_truth.append(mpimg.imread(img_path, 0))

print("Original data length: {}".format(len(images)))
print("Original gt data length: {}".format(len(ground_truth)))

final_train_data = []
final_target_train = []
for i in tqdm(range(len(images))):
    final_train_data.append(images[i])
    final_train_data.append(np.fliplr(images[i]))
    final_train_data.append(np.flipud(images[i]))

    for std in range(1, 4):
        final_train_data.append(random_noise(images[i], var=(std / 10) ** 2))

    for degree in range(1, 36):
        final_train_data.append(rotate(images[i], angle=(degree * 10), mode='wrap'))

    final_target_train.append(ground_truth[i])
    final_target_train.append(np.fliplr(ground_truth[i]))
    final_target_train.append(np.flipud(ground_truth[i]))

    for std in range(1, 4):
        final_target_train.append(random_noise(ground_truth[i], var=(0 * std / 10) ** 2))

    for degree in range(1, 36):
        final_target_train.append(rotate(ground_truth[i], angle=(degree * 10), mode='wrap'))

print("Augmented data length: {}".format(len(final_train_data)))
print("Augmented gt data length: {}".format(len(final_target_train)))

# %%

print(len(final_target_train), len(final_train_data))

create_dir_if_not_exist(AUG_PROC_IMAGE)
create_dir_if_not_exist(AUG_PROC_GT)

for i in range(len(final_train_data)):
    name_string = '{}'.format(i + 1).zfill(5)
    imageio.imwrite(AUG_PROC_IMAGE + name_string + '.png', final_train_data[i])


print(final_target_train[i].shape)

for i in range(len(final_target_train)):
    final_target_train[i] = final_target_train[i][:, :, :1]

print(final_target_train[i].shape)

for i in range(len(final_target_train)):
    name_string = '{}'.format(i + 1).zfill(5)
    imageio.imwrite(AUG_PROC_GT + name_string + '.png', final_target_train[i])

