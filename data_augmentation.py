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


GT = "/Users/mattdevries/Documents/GitHub/Pytorch-UNet/data/IOSTAR/GT/"
image = "/Users/mattdevries/Documents/GitHub/Pytorch-UNet/data/IOSTAR/image/"

images = []

for img_path in sorted(glob.glob(image + '/*.png')):
    print(img_path)
    images.append(mpimg.imread(img_path, 0))

ground_truth = []

for img_path in sorted(glob.glob(GT + '/*.png')):
    print(img_path)
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


for i in range(len(final_train_data)):
    plt.imsave('image{}.png'.format(i + 1), final_train_data[i])

# %%

for i in range(len(final_target_train)):
    plt.imsave('gt{}.png'.format(i + 1), final_target_train[i])

# %%

for i in range(len(final_target_train)):
    print(final_train_data[i].shape)

# %%

aug_path = "/Users/mattdevries/Documents/GitHub/Pytorch-UNet/data/IOSTAR_Aug/images/"
augs = []
for img_path in sorted(glob.glob(aug_path + '/*.png')):
    print(img_path)
    augs.append(mpimg.imread(img_path, 0))

# %%

for i in range(len(augs)):
    print(augs[i].shape)

# %%


for i in range(len(final_train_data)):
    imageio.imwrite('image{}.png'.format(i + 1), final_train_data[i])



print(final_target_train[i].shape)

# %%

for i in range(len(final_target_train)):
    final_target_train[i] = final_target_train[i][:, :, :1]

# %%

print(final_target_train[i].shape)

# %%

for i in range(len(final_target_train)):
    imageio.imwrite('gt{}.png'.format(i + 1), final_target_train[i])

# %%